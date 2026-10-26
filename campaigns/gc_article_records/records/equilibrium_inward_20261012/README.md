# Refined inward equilibria and degree-6 HDG checks — 2026-10-12

Four new accepted equilibria are stored in the [reusable archive](../../equilibria/README.md).
Their bands lie slightly inward from the four states selected for the article's
2026-10-10 smoke checks. The old states and their evidence are retained.

![Previous bands above, refined inward bands below](inward_comparison.png)

## Equilibrium construction

All fine solves used **8 MPI ranks with PETSc MUMPS**, one numerical-library
thread per rank, and CG2 potential and P4 torsion. The full-atlas cases also
use a CG3 recovered gradient; ITER does not construct that auxiliary field.
The installed MPICH environment already supplied both `mpirun` and `mpiexec`;
no installation or AMGX build was needed. Actual factor-backend, rank binding
and loaded-library thread counts are in each companion certificate. Actual
parallel CPU use is recorded in `star_cpu_activity.json` and `launch.json`.

The interior target size is 0.003 on the unit-scale shapes and 0.010 on ITER,
compared with 0.004 and 0.015 previously. All original affine boundary segments
are retained; boundary subdivision does not change the domain. Warm starts
sample the accepted CG2 polynomial at new CG2 nodes, followed by fresh Newton
solves and fresh certificates. No geometric extrapolation was required.
Width and smoothing stay fixed: delta=0.001, epsilon=0.00008 (ITER: 0.008,
0.00064). Increasing the midpoint moves the band inward.

| Geometry | CG triangles | CG potential DOFs | Midpoint | Previous → new mean wall clearance | New mean thickness (change) | Certified dual PDE residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Star | 828,436 | 1,660,699 | 0.0028 | 0.047422 → 0.055006 | 0.039193 (+0.171%) | 6.200e-16 |
| Ugly Pac-Man | 613,674 | 1,231,461 | 0.0025 | 0.042088 → 0.050517 | 0.039539 (+0.091%) | 1.680e-13 |
| ITER | 714,164 | 1,431,671 | 0.0224 | 0.131376 → 0.151256 | 0.109428 (-0.064%) | 4.773e-16 |
| Horseshoe | 720,448 | 1,445,757 | 0.0028 | 0.049337 → 0.057230 | 0.039152 (-0.007%) | 1.016e-15 |

Clearance and thickness are torsion-flow arclength measurements for star,
Ugly Pac-Man and horseshoe, and arc-weighted Euclidean contour measurements
for ITER. Percent changes compare the same geometry and definition. ITER's
certificate remains **PDE_AND_CLOSED_CONTOURS_VERIFIED**; the other three pass
all selected Equiband guards. No claim is made about an ITER flow atlas.

The first Ugly Pac-Man attempt failed the auxiliary atlas resolution gate
before Newton: resolved torsion fraction 0.4990234375 versus required 0.5.
Tightening ray integration tolerance from 1e-10 to 1e-11 raised it to 0.584961;
the physical parameters and acceptance gate were unchanged. The failed log
and input are retained. This fixed-window Newton catalog does not optimize
Jaccard/threshold losses; its PDE and geometry checks are not a claim about
stabilization of those optimizer functionals.

The first fine ITER setup exposed a separate issue: PDE-only initialization
still asserted a single torsion critical point and rejected a near-corner root.
The corrected PDE-only path selects the largest positive interior stationary
value without asserting uniqueness. Full flow-atlas runs retain the default
uniqueness gate, and unresolved critical cells remain rejected in both modes.
The failed attempt, regression check, and distinct producer source snapshot
used for the ITER retry are retained.

## HDG preservation

Each new state was imported through the normal Hybridge guiding-center runner.
Density is the L2 projection of W(phi_CG2) evaluated at quadrature points;
the potential and raw drift come from a fresh HDG Poisson solve. Both SI Euler
and SI BDF2 (Euler startup) took four steps with dt=0.05 to T=0.2, with p=6,
18 Duffy points per coordinate and a 22-point projection sensitivity check.
The target interior sizes are 0.010 and 0.035 (ITER), with the same polygonal
wall. This is a larger short preservation check, not a long-time study.

Numba assembly, host workers and PyPardiso were set and verified at 16 threads;
processes were bound to cores 8–23 while MUMPS used cores 0–7. The native HDG
runner uses shared-memory assembly; it was not duplicated under mpiexec.
Local Poisson Schur LU factors were cached. No KKT correction, clipping,
compatible-velocity reconstruction or HDG equilibrium polish was applied.

Here e_phi is the initial HDG/CG potential mismatch, s_h the projected
semilinear defect, r_h the normalized mass-inverted upwind HDG residual
(inverse time), and d_rho/d_phi the relative drifts from the initial HDG fields.

| Geometry | HDG triangles | e_phi | s_h | r_h | Euler d_rho(T) | BDF2 d_rho(T) | BDF2 d_phi(T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Star | 80,106 | 3.581e-06 | 8.552e-05 | 8.089e-03 | 1.159e-04 | 1.240e-04 | 1.106e-06 |
| Ugly Pac-Man | 61,200 | 5.921e-05 | 5.627e-04 | 8.380e-03 | 1.349e-04 | 1.432e-04 | 1.792e-06 |
| ITER | 63,208 | 6.091e-06 | 1.006e-04 | 1.873e-02 | 3.131e-04 | 3.342e-04 | 1.331e-06 |
| Horseshoe | 71,868 | 4.452e-06 | 8.382e-05 | 7.735e-03 | 1.134e-04 | 1.214e-04 | 1.375e-06 |

The maximum recorded physical Poisson relative residual is 1.781e-12;
maximum final relative mass drift is 8.764e-16; maximum initial density
projection sensitivity from quadrature enrichment is 5.859e-07.
Small sampled negative densities remain; these runs do not establish positivity.
The initial p=6 star attempt stopped before evolution because its Poisson
relative residual (~1.77e-12) exceeded a 1e-12 gate. The accepted campaign uses
an explicit 1e-11 gate, and records the actual residuals at every step.
That first attempt is retained under `star_attempt01_tolerance` in the HDG folder.

These checks substantially reduce the earlier 2.6–3.1% short density drift.
The source mesh, band placement, target mesh and polynomial degree all changed,
so this comparison does not estimate an isolated h- or p-convergence order.
T=10/T=100 preservation, separate spatial/time ladders and KKT/compatible-drift
comparisons remain pending. A small CG Newton residual alone would not have
established any of the measured HDG preservation results.

## Evidence and reproduction

- [Comparison data](comparison.json), [CG gallery](equilibrium_gallery.png),
  [CG catalog CSV](catalog.csv), and per-geometry checkpoints/certificates/logs.
- [HDG outputs](../equilibrium_preservation_p6_20261012/): all-step JSONL/CSV,
  configurations, initial/final coefficients, per-run summaries and CPU activity.
- `run_remaining.py`, `launch.json` and `campaign_notes.json` record producer
  commands; the HDG directory has its own launcher and per-run exact settings.
- [MPI parity](mpi_parity.json): the two-rank coarse ITER re-solve agrees with
  serial PyPardiso to 9.8e-17 in 513 sampled potentials, 8.7e-14 in relative mass,
  and less than 7e-17 in width/clearance. Its accepted checkpoint is also archived.
- Three small topology/collective tests pass on both MPI ranks; 14 archive tests,
  seven selected center/topology tests and 17 Hybridge import checks pass. Archive hash
  verification is recorded separately after the campaign.
- [Producer provenance](producer_provenance.json) contains the branch, HEAD,
  environment and hashes of frozen source copies. The producer is the dirty
  `diocotron-dolfinx` worktree; the consumer is the current Hybridge worktree.
  Warm-start sampling counts in certificates describe exporting rank zero's
  owned DOFs, not all MPI ranks.

Regenerate the offline CG figures from the diocotron worktree with:

```bash
DIOCOTRON_PY=/home/adelsaleh/src/.envs/diocotron-fenicsx/bin/python
"$DIOCOTRON_PY" -m projects.diocotron.studies.equilibrium_catalog.render /home/adelsaleh/src/gc-hdg-article/run_outputs/equilibrium_inward_20261012
"$DIOCOTRON_PY" -m projects.diocotron.studies.equilibrium_catalog.compare /home/adelsaleh/src/gc-hdg-article/run_outputs/equilibrium_inward_20261012 --archive /home/adelsaleh/src/gc-hdg-article/equilibria
```
