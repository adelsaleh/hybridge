# hp-AMG Hierarchy Choices and Literature

Original literature review: 2026-09-22. Status: proposed Poisson hierarchy
variants within the [hp-AMG solver family](README.md). The
[shared formalism](hp_amg.tex) combines their mathematical construction with
the implemented pMG-AMG variant and the nonsymmetric proposal.

The scope is to find and recommend a solver for the condensed HDG Poisson
trace system, allowing geometric or algebraic coarsening. No new solver
implementation, numerical validation, or performance result accompanies this
note. The initial design targets the existing two-dimensional scalar problem
with uniform polynomial degree and an SPD reduced trace matrix.

The strongest published match is **p-coarsening to degree one followed by
harmonic, agglomeration-based multigrid on the trace**. I recommend it as the
main candidate. Its face spaces map naturally to BSR; a GPU/BSR realization
would be an adaptation of the published algorithm.

## Closest published HDG method

Wildey, Muralikrishnan and Bui-Thanh (2019), *Unified geometric multigrid
algorithm for hybridized high-order finite element methods*, directly treats
condensed HDG elliptic systems. It reduces polynomial degree to one and then
coarsens through cell agglomeration.
[Paper](https://arxiv.org/html/1811.09909),
[DOI: 10.1137/18M1193505](https://doi.org/10.1137/18M1193505).

For an SPD assembled trace matrix, partition faces into aggregate-interior
faces I and aggregate-boundary faces B. Its harmonic transfer specializes to

\[
A=\begin{bmatrix}A_{II}&A_{IB}\\A_{BI}&A_{BB}\end{bmatrix},\qquad
P_h=\begin{bmatrix}-A_{II}^{-1}A_{IB}J\\J\end{bmatrix},\qquad R_h=P_h^T,
\]

\[
A_c=J^T\left(A_{BB}-A_{BI}A_{II}^{-1}A_{IB}\right)J,
\]

where J interpolates coarse boundary traces. Interior solves are local to
aggregates. Crucially, include the interior correction

\[
T=\operatorname{diag}(A_{II}^{-1},0).
\]

The paper reports failures of some block-Jacobi configurations when this
correction is omitted. It compares face-block Jacobi, symmetric
Gauss-Seidel, point Jacobi and Chebyshev-Jacobi.
[Transfer, correction and smoother definitions, sections 3--4](https://arxiv.org/html/1811.09909).

## Recommended starting configuration

The following choices are proposed starting points for HDGFEM. The Chebyshev
block smoother, intermediate p-level, aggregation target and terminal-size
range require comparison on the actual trace systems; they are not claimed
as published optimal settings.

| Component | Recommendation |
| --- | --- |
| p hierarchy | Start with p -> 1. Compare 6 -> 3 -> 1 as an alternative. |
| p transfer | Nested Legendre-mode injection E; residual restriction E^T; Galerkin matrix E^T A E. |
| h hierarchy | Connected cell aggregates, initially around four cells per aggregate in 2D; retain linear macro-edge traces. |
| h transfer | Harmonic extension with the interior correction described above. |
| Smoother | For the existing GPU context, start with full face-block Jacobi accelerated by Chebyshev. |
| Cycle | Symmetric V-cycle, with adjoint post-smoothing and fixed local/coarse work. |
| Terminal level | Cached direct solve; initially investigate roughly 128--512 scalar unknowns. |
| BSR blocks | In 2D: p+1 on fine edges, then 2x2 on linear macro-edges. |

For p=6, the direct route has block sizes `7 -> 2 -> 2 -> ...`; the staged
route has `7 -> 4 -> 2 -> 2 -> ...`. The p steps retain the same face graph;
h steps change the number of faces. The number of h levels should follow
actual coarse dimensions and factorization cost, rather than be fixed in
advance. Four-cell agglomeration does not guarantee a fourfold reduction in
trace unknowns on an arbitrary mesh.

Start a smoother comparison with the existing degree-two block-Chebyshev
construction and balanced 1+1 sweeps. Compare degree four or additional
sweeps only against the complete cycle cost. Spectral estimates must concern
the block-preconditioned operator. A finite power iteration with a safety
factor is an estimate, not a guaranteed spectral upper bound.

Use PCG only when the reduced operator and the fixed linear preconditioner
are SPD. FGMRES is appropriate if the proposed cycle uses variable inner
work. Flexible CG does not remove the SPD requirements. Pure Neumann or
periodic Poisson needs a compatible RHS and consistent nullspace/gauge
handling throughout the hierarchy.

## Projection operators and geometry

The current reference-Legendre normalization uses a congruence

\[
\lambda=Dz,\qquad A_o=D^TAD,\qquad b_o=D^Tb,
\qquad D_{jj}=\sqrt{(2j+1)/2}.
\]

Reference-face orthonormality is not physical-face orthonormality. On a
straight edge of length L, the corresponding physical modal mass matrix is
`(L/2) I`. Reuse the existing
[normalization and modal-transfer helpers](../../../hdgfem/backends/legendre_face_bsr.py).

For a same-mesh p step, E injects the low modes and fills higher modes with
zero. `E^T A E` is the principal modal block of the already condensed
operator. It inherits the fine discretization and stabilization; it is not
a newly assembled low-order HDG operator. Eliminating the high modes to
form another Schur complement would be a different construction, involving
a generally globally coupled high-mode block.

Residual restriction differs from projection of a field. For compatible
nested coefficient spaces with mass matrices Mf and Mc, L2 projection is

\[
Q_{L^2}=M_c^{-1}P^TM_f,
\]

whereas restriction of an assembled dual residual is `r_c=P^T r_f`.
The energy projection is `(P^T A P)^-1 P^T A`. Face-length weights should
not be inserted into residual restriction without a consistent change of
operator and vector representation.

For the geometric candidate, one concrete boundary-transfer construction is

\[
J_f=M_f^{-1}C_{fE},\qquad
(C_{fE})_{ij}=\int_f\phi_i^f\psi_j^E\,ds,
\]

using a fixed macro-edge parameter, consistent physical lengths and oriented
child-edge bases. Reversing a Legendre coordinate changes the sign of odd
modes. Linear functions of macro-edge arclength restrict exactly to linear
functions on straight child segments.

A bent macro-edge needs care: two arclength modes reproduce constants but
cannot generally reproduce both physical x and y. Prefer splitting strongly
bent or branched interfaces; coordinate-trace enrichment is an alternative
with larger or varying block sizes. Disconnected interfaces, holes and
boundary tags also constrain aggregation. Affine reproduction on arbitrary
agglomerates must be checked, not inferred from results on nested simplices.

## Related literature and limits of applicability

| Reference | Contribution and qualification |
| --- | --- |
| Lu, Rupp & Kanschat (2022), [Analysis of injection operators in geometric multigrid solvers for HDG methods](https://arxiv.org/pdf/2104.00118), [DOI](https://doi.org/10.1137/21M1400110) | Stability and reproduction of conforming linear traces; interpolation, local reconstruction and combined operators for new fine faces. Uniform h convergence assumes mesh/local-solver conditions, elliptic regularity and stabilization restrictions. It does not establish arbitrary-agglomerate or p-uniform convergence. |
| Gopalakrishnan & Tan (2009), [A convergent multigrid cycle for the hybridized mixed method](https://web.pdx.edu/~gjay/pub/hmg.pdf), [DOI](https://doi.org/10.1002/nla.636) | A convergent hybridized-mixed multigrid method using a conforming auxiliary hierarchy. Supports considering a continuous P1 auxiliary correction, with formulation-specific transfer and convergence assumptions. |
| Fu & Kuang (2023), [Optimal geometric multigrid preconditioners for HDG-P0 schemes](https://arxiv.org/abs/2208.14418) | Optimal geometric multigrid through a special quadrature-based HDG-P0/Crouzeix-Raviart equivalence. This does not automatically describe the constant-mode principal block of a higher-order HDG matrix. |
| Fu & Kuang (2024), [hp-Multigrid preconditioner for a divergence-conforming HDG scheme](https://link.springer.com/article/10.1007/s10915-024-02568-4), [manuscript](https://arxiv.org/pdf/2303.06762) | Lowest-order auxiliary correction, harmonic transfer stabilization and vertex-patch block Jacobi/GS smoothers. Their divergence-conforming flow formulation and kernel argument differ from scalar Poisson. |
| Muralikrishnan, Bui-Thanh & Shadid (2020), [A multilevel approach for trace system in HDG discretizations](https://arxiv.org/abs/1903.11045), [DOI](https://doi.org/10.1016/j.jcp.2020.109240) | Another direct HDG approach combining nested-dissection/domain-decomposition ideas with enriched multilevel trace spaces. Worth considering if simple low-order macro-interface spaces miss significant smooth error. |

These references suggest stronger element/vertex-patch smoothing when
necessary. Patch corrections should use the appropriate assembled principal
submatrices and symmetric weighting if an SPD cycle is required. They do
not establish that wider patches are always faster than face-block smoothing.

A further comparison candidate is p-to-one followed by a conforming P1
auxiliary space on the original triangulation. Prolongation takes the
continuous vertex basis onto oriented face coefficients. Begin with the
Galerkin auxiliary matrix `P^T A_1 P`; replacing it with a separately
assembled P1 stiffness matrix requires an equivalence argument. Scalar AMG
on this space is reasonable if scalar coarse levels are acceptable.

## Algebraic candidate

The recommended algebraic research alternative is

\[
p\longrightarrow1\longrightarrow
\text{face aggregation with constrained energy-minimizing interpolation}.
\]

Aggregate whole faces, use block relaxation, and construct interpolation to
represent the physical constant trace plus selected smooth-error candidates.
In the package's reference-normalized modal coordinates, a physical constant
has coefficients `(sqrt(2), 0, ...)` per face, up to a global normalization.
It is not an all-ones coefficient vector. With physical-face orthonormal
coordinates the first coefficient would instead depend on face length.
For Dirichlet problems this is a candidate for interior smooth error, not an
exact global null vector; homogeneous error boundary conditions still apply.

Dobrev, Kolev, Lee, Tomov & Vassilevski (2019), *Algebraic hybridization and
static condensation with application to scalable H(div) preconditioning*,
support AMG on hybridized systems and analyze their scalar-like low-energy
trace space. Their hybridized RT results do not directly validate arbitrary
stabilized HDG aggregation.
[Paper](https://arxiv.org/html/1801.08914),
[DOI: 10.1137/17M1132562](https://doi.org/10.1137/17M1132562).

Olson & Schroder (2011), *Smoothed aggregation multigrid solvers for
high-order discontinuous Galerkin methods for elliptic problems*, develops
block relaxation and energy-minimizing interpolation for DG Poisson. A single
weighted-Jacobi smoothing of tentative interpolation is inadequate in their
high-order setting. The application to condensed HDG traces remains an
adaptation because their systems use DG volume unknowns.
[Author manuscript](https://lukeo.cs.illinois.edu/files/2011_OlSc_hodg.pdf).

For HDGFEM, a proposed construction would rank-test local candidates, fit
those candidates in tentative interpolation, then minimize interpolation
energy on a bounded sparsity pattern subject to `P N_c = N_f`. Retain
`R=P^T` and `A_c=P^T A P`. Physical x/y traces or independently relaxed test
errors are possible enrichments. Neither their number nor the aggregation
strength rule is settled by this review.

Algebraic coarse blocks represent retained aggregate candidates. Their sizes
need not equal p+1: one candidate yields scalar blocks, while several
independent candidates give larger blocks. Preserve mathematical rank rather
than adding dependent columns to keep a preferred BSR block size. Variable
ranks require grouped or variable-block storage.

## Relation to existing HDGFEM studies

The [shared hp-AMG formalism](hp_amg.tex) describes the native pMG-AMG
variant, whose [backend](../../backends/face_hp_mg_pcg.md) already provides modal
p-coarsening, BSR operators and block-Chebyshev smoothing, followed by scalar
AMGX at p=0. The most consequential new direction is the degree-one coarse
trace space and its h-transfer.

The [hierarchy BSR feasibility study](../../research/solver_studies/hybrid_hierarchy_bsr_feasibility_2026_09.md)
reports unfavorable projections for repacking the existing scalar coarse
hierarchy into artificial blocks. The
[ASM study](../../research/solver_studies/asm_pp_bsr_opportunities_2026_09.md) reports that the tested wider
finest-level patches cost more than they saved. These are specific retained
results, not a rejection of every block hierarchy or patch method.

A future harmonic hierarchy would have changing face counts and globally
rectangular transfers, requiring capabilities beyond the present same-graph
p-cycle. Schur elimination introduces boundary coupling within aggregates,
so genuine face blocks still need fill and memory accounting. In 3D,
triangular face blocks contain `(p+1)(p+2)/2` modes; variable p and 3D
macro-face spaces require separate design work.

The [general nonsymmetric pMG-AMG proposal](../../research/solver_studies/nonsymmetric_pmg_design_2026_09_22.md)
addresses a different operator class. SPD Poisson conclusions and the
transpose restriction used here should not be transferred to nonsymmetric
transport without the corresponding analysis.

## What remains unestablished

The literature supports the candidates, but does not establish which is
fastest for HDGFEM's matrices or give a blanket hp-robustness guarantee for
the proposed BSR realization. A later comparison would need identical fine
operators, boundary conditions, stabilization, RHS, initial guesses and true
residual targets. It should separate setup, repeated-solve cost, hierarchy
memory and iteration counts across mesh sizes and polynomial degrees.

Small assembled-system checks would need to cover orientation, boundary
maps, constant preservation, applicable affine reproduction, Galerkin
identities, nullspaces, and the symmetry/positivity of the complete cycle.
They have not been performed for a new solver in this literature study.
No builds, compilation, simulations or time integration were run for it.
