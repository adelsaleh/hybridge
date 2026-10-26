# Smaller-smoothing degree-six equilibria

Requested 2026-10-16: rerun the four latest DOLFINx bands with smaller smoothing,
using the already computed potentials as initial data, and leave the work in
the background. The value 0.005 denotes epsilon/delta, reduced from 0.08.
Absolute epsilon is 5e-6 on star, Ugly Pac-Man and horseshoe, and 4e-5 on ITER.
The window widths remain 0.001 and 0.008 respectively; target placement is D_T=0.8.

`seeds.json` identifies the four accepted refined CG6 archive entries.
The existing 215,246--264,268-triangle meshes and affine walls are retained.
The continuation halves epsilon/delta through 0.04, 0.02, 0.01 and 0.005,
always initializing Newton with the preceding complete potential.
Nonlinear quadrature degree is 32 for the first two stages, 48 for the last two,
followed by a degree-64 quadrature check at the target smoothing. Auxiliary
torsion/recovery quadrature remains at its accepted value (32, or 24 for ITER).
Each stage rechecks the PDE, threshold crossings and contour topology and fits
the midpoint to the same placement target before exporting and archiving.

The first stages use eight MPI ranks, MUMPS, and one numerical-library thread per rank.
At the user's request, the next rank trial uses twelve MPI ranks; the current
eight-rank worker is allowed to finish. `switch_ranks.py` pauses only the old
coordinator, waits for that worker to exit, and resumes the launcher from accepted
stages. It records its process group in `rank_switch.driver.json` and writes
the resumed driver's output to `driver_12ranks_stdout.log`. The original
`schedule.json` records the initial eight-rank plan; `rank_control.json` controls
subsequent solves, and each execution and accepted manifest records actual ranks.
If a changed MPI partition invalidates the auxiliary flow atlas, the launcher
retains the failed attempt and retries that geometry with its accepted eight-rank
partition, keeping every geometric guard. No speedup is assumed before timing.

The domains run sequentially. Completed stages skip on resume; incomplete
attempts are retained and restarted from the preceding accepted archive.
Each solve records actual MPI activity and peak summed rank memory in `execution/`.
The process group stops if available system memory falls below 8 GiB.

Launch from this directory:

```bash
/home/adelsaleh/src/.envs/diocotron-fenicsx/bin/python prepare.py
setsid nohup /usr/bin/python3 launcher.py > driver_stdout.log 2>&1 < /dev/null &
```

`driver.json` records the process-group id and stop command. `active.json` points
to the current solve log. `completed.json` or `failed.json` records how the
campaign ended. `selected.json` and the `accepted/` links identify accepted
target-epsilon states; incomplete stages never appear in the density gallery.
`RESULTS.md`, `density_bands.png/.pdf`, and individual density/potential PNGs
are regenerated as target fields finish. All successful stages are stored in
the shared `../../equilibria` archive.

Horseshoe's first 0.08-to-0.04 reduction reached the 40-iteration limit: its
algebraic residual was still decreasing, from 6.1e-5 to 1.3e-6. The failed attempt
is retained. `horseshoe_recovery/launcher.py` waits for the original campaign's
quadrature checks to finish, then runs only horseshoe with intermediate ratios
0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005 and its quadrature-64 check.
`continuation_overrides.json` records this schedule. The main launcher supports
`--geometries`, retains already selected states, and skips accepted stages.
`recovery_pending.json` keeps the article updater waiting until this retry ends.
The recovery records its own process group and completion marker; it never runs
MPI solves concurrently with the original campaign.

The first smaller-step recovery also stopped at 40 iterations (ratio 0.06):
its algebraic residual decreased from 3.2e-5 to 3.1e-7. On 2026-10-17, the
recovery was restarted from the accepted ratio-0.08 potential with initial
ratios 0.075, 0.07, 0.06, 0.05 and the remaining finer stages. Horseshoe now
allows at most 120 Newton iterations per solve. The dual-residual and Newton
convergence tolerances are unchanged. Earlier configurations, driver markers
and logs are preserved under `horseshoe_recovery/previous_attempt_*/`.
The article updater explicitly reports missing target geometries and retains
the current figure if a recovery fails; it has been restarted for this attempt.

The thinner transitions need fresh accuracy checks. The quadrature comparisons
do not establish spatial mesh convergence. The earlier ITER 1024-trajectory
sampling limitation is also retained; these runs use its accepted 256-seed atlas.
No HDG transfer, preservation test, or guiding-center time integration is run.
The user authorized direct replacement in Figure 17 after completion.
`article_update/launcher.py` waits for the campaign completion marker, requires
accepted target-smoothing fields on all four geometries, then selects the final
archived fields, regenerates the figure and source table, updates the smoothing,
solver and accuracy statements, and rebuilds the article in its own build directory.
The PDF is installed as `../../main.pdf` only after a successful build. This
updater performs no PDE solve and does not alter the running equilibrium driver.
It preserves the preceding PDF, plots, source selection, table and section text
under `article_update/reference_before_update/`, with a SHA256 manifest.
Its `driver.json`, `driver_stdout.log`, `completed.json` and `failed.json` are
inside `article_update/`. The updater uses its own detached process group.
