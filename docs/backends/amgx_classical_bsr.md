# AMGX Classical AMG With A BSR Fine Operator

This document describes the experimental AMGX path used by HDGFEM when the
condensed HDG trace operator is uploaded in block sparse row (BSR) format and
the selected preconditioner is classical AMG. It records the maintained
algorithm and storage lifecycle. Dated performance evidence belongs in the
[August 2026 solver study](../research/solver_studies/classical_amg_bsr_2026_08.md).
The software ownership boundary and comparison with the independent HDGFEM
face-block p/h-multigrid path are defined in the
[BSR and AMGX dependency map](bsr_amgx_dependency_map.md).

Three implementations are maintained. The default is a hybrid
fine-BSR/scalar-hierarchy algorithm. The opt-in `block_graph_identity` and
`block_graph_dense` modes retain BSR matrices and transfers throughout the
hierarchy, using a compact scalar block graph only during setup. The identity
mode lifts scalar weights as `w_ic I_b`; the dense mode smooths those weights
into block-valued interpolation while enforcing `sum_c P_ic = I_b`.

## Fine Trace Operator

For polynomial degree `p`, one HDG face contributes a dense block of size

\[
b = p + 1.
\]

The reduced trace operator is stored as

\[
A_0 = [A_{ij}], \qquad A_{ij}\in\mathbb{R}^{b\times b},
\]

with one BSR row per free global face. The raw-CUDA assembly path emits this
block graph and its dense block values directly; it does not replace the
existing direct-CSR path.

## Hybrid Hierarchy Construction

Classical AMGX coarsening is scalar. During setup only, the BSR fine operator
is expanded coefficient-exactly into a scalar CSR view

\[
\mathcal E(A_0)_{ib+r,\,jb+s} = (A_{ij})_{rs},
\qquad 0\le r,s<b.
\]

The expansion changes storage and graph indexing only. It does not lump,
average, or otherwise alter coefficients. Existing classical strength,
PMIS/aggressive selection, D2 interpolation, and Galerkin products then act on
`E(A_0)` without changing their scalar definitions:

\[
A_{\ell+1}=R_\ell A_\ell P_\ell.
\]

The temporary scalar expansion is released after selection, including a
rejected candidate level. It is recreated only when needed to build an
accepted first coarse operator and is released again afterwards. The retained
fine operator remains BSR; retained transfer and coarse operators are scalar
CSR.

The initial implementation requires:

- one GPU;
- square BSR blocks;
- row-major or column-major dense blocks;
- an internal diagonal;
- index ranges representable by the configured AMGX index type.

Distributed BSR matrices and separately stored diagonal blocks are rejected
explicitly.

## V-Cycle Execution

The fine-level matrix-vector product and Jacobi-L1 smoother operate directly
on BSR storage. With `jacobi_l1_scalar_rows_for_blocks=1`, each scalar row
inside a block row uses the same L1 diagonal as coefficient-exact scalar CSR:

\[
d_i = \operatorname{sign}(a_{ii})\sum_j |a_{ij}|,
\]

where a nonnegative diagonal selects the positive sign. Both dense-block
layouts are supported.

At the first restriction, the contiguous fine vector is temporarily exposed
with scalar block metadata so that scalar `R_0` consumes the same coefficients
without a data copy. Coarse right-hand-side and correction work vectors take
their block dimensions from the next-level scalar operator. Prolongation uses
scalar `P_0`, applies the correction coefficient-wise, and restores the fine
block metadata before post-smoothing.

Consequently, no fine CSR matrix is retained or used by fine smoothing during
the solve. The V-cycle is nevertheless not CSR-free: restriction,
prolongation, and all coarse operators are scalar CSR.

## Configuration

The validated classical configuration enables scalar-row Jacobi-L1 for the
BSR fine operator:

```json
{
  "solver": {
    "preconditioner": {
      "smoother": {
        "solver": "CHEBYSHEV",
        "preconditioner": {
          "solver": "JACOBI_L1",
          "jacobi_l1_scalar_rows_for_blocks": 1
        }
      }
    }
  }
}
```

HDGFEM's checked-in example is
`configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`. The option is
opt-in so legacy AMGX block-Jacobi behavior remains unchanged.

## Required Invariants

Implementations and tests must preserve these invariants:

1. BSR and CSR represent the identical reduced HDG operator and boundary
   elimination.
2. Scalar expansion preserves every coefficient and face-basis ordering.
3. Fine vector storage is not copied merely to change block metadata.
4. Coarse work-vector dimensions match the next-level operator, not the fine
   operator.
5. The independently evaluated physical residual, rather than AMGX telemetry
   alone, decides solve acceptance.
6. Temporary setup expansions are released on accepted, rejected, and failed
   hierarchy paths.

## Pure-BSR Block-Graph D2 Algorithm

The first pure-BSR implementation is selected explicitly with
`classical_bsr_hierarchy=block_graph_identity`. The coefficient-exact hybrid
algorithm remains the default. Dense block interpolation is a later mode, not
an implicit change to this one.

Every retained hierarchy object keeps the fine face block size `b`:

\[
A_\ell\in\operatorname{BSR}(b,b),\quad
P_\ell,R_\ell\in\operatorname{BSR}(b,b),\quad
A_{\ell+1}=R_\ell A_\ell P_\ell.
\]

### Temporary Scalar Block Graph

Setup constructs a scalar matrix `G_l` with one scalar row per block row and
one scalar nonzero per BSR block. Its graph is exactly the BSR graph. For an
off-diagonal block,

\[
g_{ij}=-\lVert A_{ij}\rVert_F,\qquad i\ne j,
\]

and its diagonal is the positive off-diagonal row sum,

\[
g_{ii}=\sum_{j\ne i}\lVert A_{ij}\rVert_F.
\]

An isolated row uses `g_ii = 1`. This produces the sign pattern expected by
classical strength and is invariant under orthogonal changes of basis inside
a block. `G_l` is setup-only: it is used by the existing strength, PMIS, and
D2 implementations, then released. It is never used for smoothing or a
V-cycle matrix-vector product.

### Identity-Lifted Interpolation

D2 applied to `G_l` gives scalar weights `w_ic`. The retained prolongation and
restriction use the same block graph but lift each scalar weight to an
identity block:

\[
(P_\ell)_{ic}=w_{ic}I_b,
\qquad
R_\ell=P_\ell^{\mathsf T}.
\]

Thus `P_l` reproduces all `b` componentwise constant block modes whenever the
scalar D2 row weights sum to one. Fine and coarse vectors keep block dimension
`b`; restriction and prolongation require no scalar metadata reinterpretation.

### Weighted BSR Galerkin Product

The coarse operator is accumulated directly in BSR. With scalar identity-lift
weights, each dense coarse block is

\[
(A_{\ell+1})_{cd}
=\sum_i\sum_j w_{ic}\,A_{ij}\,w_{jd}.
\]

The implementation uses a symbolic pass over the block graphs followed by a
numeric pass over dense `b x b` values. It may consume the compact scalar
weight graph while building `A_(l+1)`, but it must emit a canonical BSR matrix
with sorted, duplicate-free block columns and an internal diagonal. It must
not expand `A_l`, `P_l`, `R_l`, or `A_(l+1)` coefficient-by-coefficient into
CSR.

The same construction is repeated from each retained coarse BSR operator.
Consequently all smoothing, residual products, restriction, prolongation, and
coarse-grid operations remain block operations throughout the V-cycle.

### Configuration And Safeguards

```json
{
  "solver": {
    "preconditioner": {
      "classical_bsr_hierarchy": "block_graph_identity"
    }
  }
}
```

The first implementation is single-GPU, requires square blocks with an
internal diagonal, and supports D2 interpolation. Unsupported distributed,
rectangular-block, external-diagonal, or alternative-interpolator cases fail
explicitly rather than falling back silently. Hierarchy reuse must retain BSR
transfer graphs and recompute their scalar weights and Galerkin block values
when matrix values change.

The accompanying AMGX patch also generalizes DenseLU's BSR-to-dense
conversion. One warp still owns a block row, but its lanes traverse the dense
block coefficients in strides of 32. Without that stride, only the first 32
coefficients were copied: 6x6 and 7x7 coarse blocks (HDG degrees 5 and 6) were
truncated and cuSOLVER received a singular dense coarse matrix. This is a
DenseLU compatibility correction and does not change the identity-lifted
Galerkin definition.

### Current Validation Status

The pre-fix bounded tests appeared to show a p=6 PCGF breakdown, but those
results were invalidated by the multilevel correction overrun described below.
After repairing the overrun, both PCGF and FGMRES converge for every tested
block size. PCGF is the preferred outer solver for this symmetric Poisson
operator because it requires fewer iterations and less Krylov storage:

| validation | PCGF result |
|---|---|
| 99,896 triangles, p=1--6 | 312--500 iterations; relative residual at most `9.869e-9` |
| 124,831 triangles, p=1--6 | 341--592 iterations; relative residual at most `9.662e-9` |
| 150,209 triangles, p=1--6 | 355--654 iterations; relative residual at most `9.314e-9` |
| paired 150,209-triangle p=6 | 590 iterations, relative residual `4.302e-9`, coefficient difference from CSR `3.962e-7` |

Use
`configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_identity_bsr.json`
for the validated PCGF path. The hierarchy remains experimental because its
identity-lifted coarse correction is weak, not because it is CG-incompatible.

On the tighter p=6 sample, pure-BSR setup/solve took `0.052/0.210 s`, versus
`12.388/0.018 s` for coefficient-expanded CSR. These were initial bounded
runs; the superseding 99,896/124,831/150,209-triangle degree sweep is recorded
in the [August 2026 solver study](../research/solver_studies/classical_amg_bsr_2026_08.md#pure-bsr-identity-lift-follow-up-2026-08-19).

#### Multilevel correction-size defect and fix

The first three-level implementation failed during a later restriction or
prolongation because `prolongateAndApplyCorrection` passed a scalar entry count
as the `size` argument of `axpby`. AMGX's vector BLAS interprets that argument
as a vector-block count and multiplies it by `x.get_block_size()`. For block
size `b`, the pure-BSR branch therefore processed `n_block_rows*b*b` entries
instead of `n_block_rows*b`, reading and writing beyond the correction vectors.
The writes eventually reached pooled transfer storage and appeared as corrupted
`P` or `R` column indices; transfer construction itself was not the cause.

The corrected branch passes `P.get_num_rows()`, retains the vector block size
`b`, and checks the resulting `owned_size*block_size` extent against both
vectors before applying the correction. All block-graph CUDA kernels also use
AMGX's configured stream rather than assuming stream zero.

The formerly failing 6,369-triangle p=6 case now converges in 120 FGMRES
iterations with relative residual `6.833e-9`. CUDA Initcheck and Memcheck each
report zero errors. The subsequent 72-run FGMRES/CSR sweep and 18-case PCGF
sweep completed without a transfer-structure or memory failure.

Validation must check:

1. `P` maps every blockwise constant component to the same component.
2. Explicit scalar expansion of the identity-lifted `RAP` agrees coefficient
   by coefficient with the weighted BSR Galerkin result on small matrices.
3. Every retained matrix and transfer reports block dimensions `b x b`.
4. CSR, hybrid BSR, and pure BSR solves meet the same independently evaluated
   physical residual.
5. No coefficient-expanded CSR allocation occurs in pure-BSR setup or solve.

### Dense Block Interpolation Mode

The opt-in `block_graph_dense` mode keeps the Frobenius block graph, PMIS/D2
coarse selection, and the truncated D2 block support. It changes only the
numeric transfer blocks. Starting from

\[
(P_0)_{ic}=w_{ic}I_b,
\]

one damped block-Jacobi energy step forms

\[
\widetilde P=P_0-\omega D^{-1}AP_0,
\qquad D=\operatorname{blockdiag}(A).
\]

The implementation uses a pivoted block solve for every diagonal block and
does not form `D^{-1}A`. Entries generated outside the existing D2 support are
discarded deliberately, so identity and dense modes have the same block
sparsity and can be compared without a graph-density confounder.

The smoothing step alone need not preserve the `b` componentwise constant
modes. For each fine block row define

\[
E_i=I_b-\sum_c \widetilde P_{ic},
\qquad
\alpha_{ic}=\frac{w_{ic}}{\sum_d w_{id}},
\]

and apply the local correction

\[
P_{ic}=\widetilde P_{ic}+\alpha_{ic}E_i.
\]

If the scalar row sum is numerically zero, `alpha_ic` is uniform over the
retained row support. Consequently,

\[
\sum_c P_{ic}=I_b
\]

to the configured validation tolerance. This is an explicit block
near-nullspace constraint, not an assumption inherited from scalar D2.

Restriction is constructed as the actual conjugate block transpose `R=P^*`.
The coarse operator is the exact dense-block Galerkin product

\[
(A_{\ell+1})_{cd}
=\sum_i\sum_j P_{ic}^{*}A_{ij}P_{jd}.
\]

The symbolic coarse graph is still computed from the compact scalar block
graphs, but the retained `P`, `R`, `A`, and every V-cycle product remain
BSR. Dense transfers use the generic BSR SpMV path;
`block_graph_identity` retains its specialized diagonal-only transfer kernel
and its original weighted Galerkin definition.

Configuration:

```json
{
  "classical_bsr_hierarchy": "block_graph_dense",
  "block_graph_dense_smoothing_weight": 0.6666666666666666,
  "block_graph_dense_smoothing_steps": 1,
  "block_graph_dense_constraint_mode": "additive",
  "block_graph_dense_pivot_tolerance": 1e-12,
  "block_graph_dense_constraint_tolerance": 1e-8
}
```

The validated baseline uses one smoothing step and `additive` correction,
which preserves its original numerical definition. Additional true block
smoothing steps reuse the preceding dense interpolation blocks in
`P <- P - omega D^-1 A P`; they do not fall back to the scalar tentative
weights. The opt-in `right_normalize` constraint replaces additive correction
with

\[
P_{ic} \leftarrow P_{ic}\left(\sum_d P_{id}\right)^{-1},
\]

applied after every smoothing step. Both modes verify
`sum_c P_ic = I_b`, retain exact `R=P^*`, and form the same dense BSR Galerkin
operator. The right-normalized path requires each block row sum to be
nonsingular and reports the first failure rather than silently regularizing it.

The implementation is single-GPU, requires square blocks, D2 interpolation,
an internal nonsingular block diagonal, and currently rejects
`structure_reuse_levels > 0` and `aggressive_levels > 0` explicitly. The
aggressive D2 pass can leave a fine block row with no interpolation support,
which is incompatible with the enforced `sum_c P_ic = I_b` constraint; support
completion is required before that combination can be enabled. Pivot and
constraint tolerances use a floor of 100 times the matrix precision epsilon,
so the defaults remain meaningful in both float and double builds. The
reference dense Galerkin kernel prioritizes a verifiable `P^* A P` result;
setup profiling will determine whether its matrix products need a tiled or
library-backed optimization.

Use
`configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_dense_bsr.json`
for PCGF/Chebyshev validation. On the 4,658-triangle radius-5 disk at p=6 and
a `1e-10` relative tolerance, dense pure BSR converged in 27 iterations versus
25 for scalar CSR. The independently reconstructed primal coefficient vectors
differed by `1.476e-10`. The BSR solve took 0.098 s versus 0.065 s for CSR,
while its compressed-pattern storage was 2.38% of CSR's. Thus coarse-space
quality is now comparable on this test; transfer and coarse-cycle throughput,
plus larger CUDA memory checks, remain to be optimized and validated. The mode
does not alter the numerical definition of `block_graph_identity`.
