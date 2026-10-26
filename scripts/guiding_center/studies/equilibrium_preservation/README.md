# Imported equilibria: transfer and short preservation checks

The standard guiding-center runner accepts `--case imported_equilibrium`.
Choose `--case-param archive=PATH --case-param equilibrium=ID` (a stable ID or
unique archive name), or `--case-param checkpoint=FILE.npz`. The original
embedded affine mesh is the default. `--case-param remesh_size=H` remeshes only
the interior, retaining every archived wall segment. The common runner's
`--mesh-size` does not replace this case-specific choice. Domain overrides and
non-zero-flux transport walls are rejected.

The portable reader accepts both `hdgfem_*_v2` and `hybridge_*_v2` checkpoints.
No DOLFINx installation is needed in the Hybridge environment. It validates
array checksums; archive selection also validates checkpoint/certificate file
hashes and matching state/window metadata. It reconstructs the full CG2
potential, evaluates the frozen nonlinear window at target quadrature points,
and L2-projects the resulting density. It does not interpolate nodal density,
clip the projection, or transfer the CG convergence certificate to HDG.
Sampling is on the host; device initialization explicitly uploads its results.
CPU execution is validated here; GPU imported-case parity remains untested.

An HDG Poisson solve initializes the potential/electric field. Initial JSONL
records contain source provenance and:

- `equilibrium_poisson_cg_relative_l2`: HDG potential versus projected CG potential.
- `equilibrium_semilinear_relative_l2`: rho versus the projected window of the HDG potential.
- `equilibrium_steady_transport_relative_l2`: mass-inverted HDG transport residual divided by the density norm, including numerical faces; units 1/time. The diagnostic supports the upwind/conflict-averaged-upwind residual operator.
- `equilibrium_projection_quadrature_relative_l2`: change from increasing the initial projection quadrature by four points per Duffy coordinate (at least 12+4 if no initial rule is specified).
- Existing `poisson_physical_rel_residual` and `diocotron_{rho,phi}_eq_relative_l2`: linear residual and evolution drift. The potential reference is the initial **HDG** Poisson solution, separately from the CG/HDG mismatch.

## Bounded smoke driver

The following runs one case at a time on CPU; it caps the mesh before assembly.
It uses the ordinary GC runner and writes configs, all-step JSONL/CSV diagnostics,
initial/final native fields, summary JSON and observed CPU activity. Outputs must
not already exist. `--dry-run` only prints settings. `--threads` sets and verifies
Numba, host NumPy workers and Pardiso (default 16); `NUMBA_NUM_THREADS` must be at
least that large at process startup. Keep OpenBLAS at one thread for small local
matrix algebra and avoid nested BLAS worker pools. Higher-order runs can select
`--poisson-cache-local-factors schur-lu` to reuse the Numba local Schur factors.
Defaults retain the original uncached behavior. `--linear-rtol` sets the checked
physical relative tolerance for both condensed solves (default 1e-12). On the
80,106-triangle p=6 star system, the first direct Poisson residual was 1.77e-12;
the refined campaign therefore explicitly requests 1e-11 and retains actual
residual measurements, including the rejected initial attempt.

```bash
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 \
NUMBA_NUM_THREADS=16 MPLBACKEND=Agg .venv/bin/python \
  scripts/guiding_center/studies/equilibrium_preservation/run_equilibrium_preservation.py \
  --archive ../gc-hdg-article/equilibria \
  --output ../gc-hdg-article/run_outputs/equilibrium_check_new \
  --equilibria star_02,uglypacman_04,iter_02,horseshoe_02 \
  --schemes si-euler,si-bdf2 --orders 4 \
  --remesh-size .04 --iter-remesh-size .15 --quadrature-1d 16 \
  --dt .05 --steps 4 --threads 16 --max-triangles 20000
```

`--original-mesh` and a deliberately increased triangle cap enable future
source-mesh studies. They were not used in the current smoke checks. SI BDF3 is
accepted by the driver but was not tested in this pass. A `--case` supplied
programmatically to `run_guiding_center_case` can reuse the loaded source/mesh
between schemes; its key must match the configuration.

## Measured status, 2026-10-10

Four representative states were evolved: star_02, uglypacman_04, iter_02 and
horseshoe_02. Twelve fine archived states plus the coarse ITER initializer all
passed import/sampled-CG2 checks. Preserving an equilibrium has **not** been
established: at T=0.2 the p=2 checks drift by 7.75–9.15% in density; the finer
p=4 checks drift by 2.60–3.10%. The p=4 semilinear mismatch is 2.32–4.60% at t=0.
Projection quadrature sensitivity falls to 2.6e-5–1.15e-4. Tight Poisson residuals
(<7.7e-14) and conserved mass do not remove those spatial defects. Negative DG
undershoots remain; no KKT limiter or HDG equilibrium correction was applied.

The original two levels used Numba's one-thread pool and Pardiso's verified
16-thread setting. After correcting the assembly thread control, one repeated
star p=4 BDF2 check with all three worker counts explicitly set to 16 agreed
within 2.9e-14 in final density coefficients. The timing records include setup
and diagnostics and are not performance benchmarks.

Evidence is in the adjacent article's `run_outputs/` directories:
`equilibrium_preservation_smoke_20261010`,
`equilibrium_preservation_refined_smoke_20261010`, and
`equilibrium_preservation_threads16_20261010`.
The article's Section 7 reports these as preliminary results; separate h/p/dt
ladders, full source-mesh and long-time preservation, compatible velocity and
KKT comparisons remain open.

Offline figure/table generation (no solve):

```bash
MPLBACKEND=Agg .venv/bin/python \
  scripts/guiding_center/studies/equilibrium_preservation/plot_equilibrium_preservation.py \
  ../gc-hdg-article/run_outputs/equilibrium_preservation_smoke_20261010/summary.json \
  ../gc-hdg-article/run_outputs/equilibrium_preservation_refined_smoke_20261010/summary.json \
  --figure ../gc-hdg-article/figures/equilibrium_preservation_smoke \
  --table ../gc-hdg-article/tables/equilibrium_preservation_smoke.tex
```


## Refined inward campaign, 2026-10-12

Four new sources were recomputed with 8-rank MPI MUMPS on 613,674–828,436 CG2
triangles. Their bands are 15–20% farther from the wall with less than 0.2%
change in mean thickness. Each passed fresh source acceptance checks; ITER
retains its PDE/three-contour certificate, without an auxiliary flow atlas.

The native CPU consumer then used p=6, 61,200–80,106 target triangles,
18-point Duffy quadrature (22-point sensitivity check), `schur-lu` local
Poisson caching, and 16 verified threads. Both Euler and BDF2 took four steps
at dt=0.05. Density drift at T=0.2 is 0.0113–0.0334%; potential drift is
1.06e-6–1.79e-6. Maximum measured Poisson relative residual is 1.781e-12,
with an explicit 1e-11 acceptance gate. Small sampled negative densities remain.
Band placement and both meshes/p changed together, so no isolated convergence
order or long-time preservation is asserted. Full measurements, attempts,
source snapshots and figures are in the
[combined report](../../../../../gc-hdg-article/run_outputs/equilibrium_inward_20261012/README.md).
