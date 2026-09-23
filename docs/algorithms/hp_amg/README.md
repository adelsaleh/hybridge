# Face-Block hp-AMG Solvers for HDG Trace Systems

The hp-AMG family combines polynomial reduction, geometric or agglomerated
h-coarsening, and algebraic multigrid on the condensed HDG trace system.
The variants share the HDG equations, modal coordinates, block storage and
multigrid correction structure; they differ in their coarse spaces, transfer
operators, smoothers and outer iterations. Here hp-AMG is the documentation
family name, not a new solver selector or a requirement that every variant
use both geometric h- and p-coarsening.

| Variant | Hierarchy | Transfers and outer iteration | Status |
| --- | --- | --- | --- |
| Native SPD pMG-AMG | Same-mesh p -> 0, then scalar AMGX | Nested Galerkin transfers, symmetric cycle, flexible PCGF | Implemented for the configurations in the backend reference |
| Poisson harmonic hp multigrid | p -> 1, then cell agglomeration and macro-edge coarsening | Harmonic extension and interior correction; PCG requires an SPD cycle | Literature-backed design |
| Poisson algebraic hp-AMG | p -> 1, then face aggregation | Constrained energy-minimizing interpolation, R=P^T, block relaxation | Proposed adaptation |
| Nonsymmetric hp-AMG | Algebraic polynomial reduction, then block/scalar AMG at an admissible degree | Independent Petrov-Galerkin transfers, nonsymmetric relaxation, right FGMRES | Proposed generalization with small-matrix algebra evidence |

The native path is historically called hp-BSR or FB-HP-MG-PCG. Its public
selector remains `solver="fb-hp-mg-pcg"`, and its outer recurrence is flexible
PCGF. It assumes an SPD reduced trace system and an appropriate SPD
preconditioner. BSR storage itself imposes no symmetry requirement. Runtime
symmetry and curvature samples are guards, not proofs of positive definiteness.

## Shared formalism and hierarchy choices

[`hp_amg.tex`](hp_amg.tex) combines the mathematical treatments:

1. **Shared HDG and multigrid formalism:** static condensation, modal coordinates,
   per-level spaces, Galerkin/Petrov-Galerkin operators, and the distinction
   between residual restriction and field projection.
2. **Implemented SPD pMG-AMG:** nested modal transfers, block-Chebyshev
   smoothing, V-cycle, conditional SPD argument, PCGF, convergence checks
   and current policies.
3. **Proposed nonsymmetric hp-AMG:** independent algebraic transfers,
   block/patch relaxation, FGMRES and nonsymmetric AMG candidates.
4. **Poisson h/p hierarchy alternatives:** harmonic agglomeration, interior
   correction, macro-edge geometry, energy-minimizing AMG and auxiliary spaces.

The [hierarchy choices and literature review](hierarchies.md) compares the
Poisson candidates, explains their BSR mapping, and distinguishes published
results from proposed HDGFEM adaptations. The original review was prepared
on 2026-09-22; its design content is maintained here with the solver family.

For the implemented variant, the p-levels keep the same face graph and end
at degree zero before scalar AMGX coarsens algebraically. The harmonic
candidate instead retains degree-one traces and changes the face graph
through agglomeration. The shared framework permits both constructions;
neither one is presented as the only meaning of hp multigrid.

## Nonsymmetric variant and its evidence

The nonsymmetric section of [`hp_amg.tex`](hp_amg.tex) specifies **right FGMRES + face-block
Petrov-Galerkin polynomial reduction + nonsymmetric AMG**. Setup uses the
condensed matrix, face blocks and polynomial grouping, without requiring a
velocity, diffusion tensor or advection/diffusion classification.

Low trace modes acquire high-mode extensions from local matrix couplings;
restriction and prolongation are independent. An unacceptable p split must
be changed or skipped, with block AMG starting at an admissible retained
degree. The note derives ideal block-elimination identities, sparse local
approximations, the cycle and original-system residual contract. It discusses
block AIR and constrained AIR as spatial algebraic hierarchy candidates.
This proposal is not an available solver selector, and SPD theory does not
establish its convergence.

- [Small-matrix evidence and implementation gaps](../../research/solver_studies/nonsymmetric_pmg_design_2026_09_22.md)
- [Reproducible algebra diagnostic](../../../scripts/dev/check_nonsymmetric_pmg_algebra.py)
- [Block AIR research](https://arxiv.org/abs/1708.06065)
- [AIR applied to HDG trace systems](https://arxiv.org/abs/2010.11130)
- [Constrained AIR across advection and diffusion](https://arxiv.org/abs/2307.00229)

## Implementation and related documentation

- [Native hierarchy, V-cycle and outer iteration](../../../hdgfem/linalg/face_hp_multigrid.py)
- [Policy and scalar AMG configuration](../../../hdgfem/linalg/face_hp_policy.py)
- [Modal BSR operators and transfers](../../../hdgfem/backends/legendre_face_bsr.py)
- [Production backend and fallback](../../backends/face_hp_mg_pcg.md)
- [HDG diffusion-reaction formulation](../diffusion_reaction/README.md)
- [Historical design and tuning evidence](../../development/plans/face_block_hp_multigrid.md)

Backend support remains governed by the
[capability reference](../../reference/backend_capabilities.md).

## Compile the combined document

From the repository root, run twice to resolve the contents and cross-references:

```sh
cd docs/algorithms/hp_amg
pdflatex -interaction=nonstopmode -halt-on-error hp_amg.tex
pdflatex -interaction=nonstopmode -halt-on-error hp_amg.tex
```

The user runs compilation. Generated PDFs are local artifacts and are not tracked.
