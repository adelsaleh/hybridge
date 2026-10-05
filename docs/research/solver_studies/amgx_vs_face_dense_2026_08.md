# AMGX versus face-dense GMRES diffusion benchmark (2026-08-13)

## Scope

This timing campaign compares the best configuration found for the current
AMGX diffusion path with the best configuration supplied by the
`gpu_gmres_precondit` face-dense branch.  It is a performance comparison, not
a new convergence study.

The common problem and discretization were:

- classical trigonometric Poisson problem on the disk centered at the origin
  with radius 5;
- 2,079, 8,196, 32,449, and 148,546 triangles, generated with Gmsh target
  sizes 0.300, 0.150, 0.075, and 0.035;
- orders p=1,...,6;
- `dub_orth` volume basis and `legacy-lagrange` trace basis;
- volume and edge quadrature counts equal to 2p;
- stabilization `tau=1`, boundary elimination, and requested relative
  residual tolerance 1e-8;
- Quadro RTX 6000, CUDA runtime 12.9, CUDA driver API 13.0.

The face-dense implementation was run unchanged from commit
`ea5ad26281f9e988194d9352399ccc4354a6633e` in an isolated worktree.  The
current master implementation was commit
`92c97fe9435fac0e58d30bb09c32ec1bf6a4b5d0`, apart from the standalone disk
runner's stale duplicate `mesh_size` argument fixed during this campaign.

The validated face-dense implementation and focused tests have since been
selectively ported into the current tree under `hybridge.linalg`,
`hybridge.backends.cupy_*`, and `hybridge.solvers.diffusion_face_dense`. The
historical timings below remain tied to the two commits named above; the
canonical current runners are
`scripts.diffusion_reaction.validate_face_dense_gpu_solver` and
`scripts.diffusion_reaction.benchmark_face_dense_primitives`.

## Solver configurations

The face-dense configuration was the branch recommendation:

- raw face-dense operator;
- fused additive Schwarz;
- degree-18 harmonic-Ritz/Leja polynomial preconditioner;
- CGS for outer restarted GMRES and CGS2 for spectral setup;
- restart 100, cuBLAS batched local inverse;
- one discarded warm-up and two measured solves; tables use medians.

The initial AMGX grid used the checked-in
`diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`: PCGF with classical
PMIS/D2 AMG, one aggressive level, and Chebyshev/Jacobi-L1 smoothing.  Its
`dense_lu_num_rows=2048` caused reproducible setup cliffs: for example, the
8,196-element (p=2) case spent 61.1 seconds in setup while PCGF needed one
iteration.

A second complete grid changed only `dense_lu_num_rows` from 2048 to 128.
It passed all 24 cases, reduced aggregate application time from 174.0 to
59.3 seconds (2.93x), and was faster on 23/24 rows.  The remaining row differed
by only 0.002 seconds.  The dense-128 variant is therefore the AMGX
configuration used below.

Each AMGX case ran in a new subprocess.  Its setup, iterate, validation, and
raw-CUDA assembly timings are exact backend timers.  The reported AMGX HDG
total is approximate because the standalone runner prints global-solve and
reconstruction headline phases to 0.1 seconds.

## Fine-mesh results

All entries below use 148,546 triangles.  `Other/local` for the face-dense
path is the remainder of its measured HDG total after global face assembly,
operator/preconditioner setup, and GMRES; it includes legacy local assembly,
field reconstruction, and related work.

| p | AMGX it | Face it | AMGX assembly | AMGX setup | AMGX iterate | AMGX HDG approx | Face assembly | Face setup | Face iterate | Face other/local | Face HDG | Face/AMGX |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21 | 74 | 0.433 s | 0.057 s | 0.076 s | 0.633 s | 4.587 s | 1.293 s | 0.668 s | 1.178 s | 7.727 s | 12.20x |
| 2 | 20 | 82 | 0.695 s | 0.066 s | 0.110 s | 0.995 s | 5.496 s | 1.500 s | 1.794 s | 2.835 s | 11.626 s | 11.68x |
| 3 | 22 | 85 | 1.173 s | 0.073 s | 0.173 s | 1.473 s | 6.690 s | 1.769 s | 2.538 s | 6.663 s | 17.660 s | 11.99x |
| 4 | 21 | 94 | 1.804 s | 0.098 s | 0.210 s | 2.204 s | 8.551 s | 2.100 s | 5.787 s | 14.380 s | 30.818 s | 13.98x |
| 5 | 19 | 94 | 2.573 s | 0.090 s | 0.250 s | 3.073 s | 11.047 s | 2.563 s | 7.695 s | 26.308 s | 47.612 s | 15.49x |
| 6 | 21 | 97 | 3.670 s | 0.103 s | 0.346 s | 4.370 s | 14.951 s | 3.003 s | 10.962 s | 46.991 s | 75.906 s | 17.37x |

Both solvers met the requested physical residual level.  Across these fine
cases, AMGX physical residuals were 4.38e-9 to 9.65e-9; face-dense residuals
were 8.06e-10 to 8.70e-9.

The primal L2 errors agree closely through p=3.  The old face-dense path then
saturates near 3.6e-6, while the AMGX/master path reaches about 1e-7.  Error
integration is not strictly identical:
the AMGX sweep uses its independent higher-order error quadrature, whereas the
old branch evaluates the error with the 2p space quadrature.  Treat this as
a trend and an integration target, not proof of an operator discrepancy.

## Crossover and conclusions

For the complete HDG path:

- face-dense wins the smallest 2,079-element cases for p=1,...,5;
- the 2,079-element p=6 case already favors AMGX by 1.44x;
- at 8,196 elements, AMGX wins for p>=2 (the p=1 ratio is near
  parity);
- at 32,449 elements, AMGX is 2.90x to 10.31x faster;
- at 148,546 elements, AMGX is 11.68x to 17.37x faster.

The face-dense global GMRES solver itself remains credible: it solved every
case without cuSPARSE or OOM, and the 148,546-element solve phase ranged from
0.668 seconds at p=1 to 10.962 seconds at p=6.  Its current integration
is dominated by the old local HDG assembly and reconstruction, and its
iteration count grows to 74--97 on the fine mesh versus 19--22 for AMGX.

The fair next integration step is therefore to preserve the face-dense
operator/GMRES/ASM-polynomial solver while replacing its old local assembly
and reconstruction with current master raw-CUDA kernels.  The benchmark should
then be repeated with identical error quadrature and matched peak-memory
instrumentation.

Raw logs and full JSON remain under
`/tmp/codex-diffusion-benchmark/`.  The compact 24-row evidence table is
stored beside this report.
