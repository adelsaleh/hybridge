# Inner band: Section 7.3 repeated with the band a few thicknesses from the wall (2026-10-21)

Request (user): "repeat the whole experiment in section 7.3 but with a band having a few more units away from
the boundary so that the diameter of triangles outside the band and near the wall are not far from the
diameter of the triangles in the inner section. before running, compute the equilibrium and show it to me. it
has to be a very well resolved equilibrium to take a big mesh and high polynomial order, and scale mpi
accordingly. once I confirm the equilibrium, we can relaunch the campaign. (take a small epsilon)"

## Design

Same undercut star and the same window width as the near-wall band of Section 7.3
(`e7_near_wall_equilibrium_20261020`: delta = 7.5e-4, m = delta, epsilon/delta = 0.02, gap 0.27 d). Only the
midpoint and the smoothing change:

| quantity | near-wall (7.3) | inner band (here) |
|---|---|---|
| delta | 7.5e-4 | 7.5e-4 |
| midpoint m | delta | 7 delta |
| epsilon/delta | 0.02 | 0.005 (the sharp four-domain sources of Fig. 14) |

Why m = 7 delta: the gap is charge-free, so the potential drop across it, m - delta/2 = 6.5 delta = 4.9e-3,
sets the distance at the local gradient. The level sets of the broad-band source at the target levels give a
first estimate: outer contour length 4.9 (the band no longer follows every undercut; the near-wall band had
L = 9.0), wall gap 0.044-0.17 (mean 0.080). The gradient at the outer edge is then about 0.069 (flux 0.34
through a contour of length 4.9), the band about 0.021 thick, so the gap is about 2 d at the undercut corners
and 4 d on average. Pilot (stage A) and medium (stage B) stages refine this before the big mesh is built.

Edge widths at epsilon/delta = 0.005 (10-90 % of W): outer about 4.4 epsilon/|grad phi| = 2.4e-4; inner about
5 sqrt(epsilon) = 0.01, because the gradient vanishes at the inner edge (phi is quadratic there) so the inner
width scales as the square root of epsilon, not linearly. The small epsilon therefore sharpens the inner edge
only slowly: 0.49 d at 0.02 on the near-wall band, about 0.45 d here (thinner band).

Meshes (`make_source_mesh.py`: separate sizes at the outer and inner contours, bulk 0.015 so the cells in the
gap next to the wall match the interior cells, grading 0.3, corner size 0.003):

| mesh | outer / inner size | purpose |
|---|---|---|
| mesh_pilot | 0.004 / 0.004 (bulk 0.025, buffer 0.012) | CG3 homotopy at epsilon/delta 0.08, contours for the next mesh |
| mesh_medium | 7.5e-4 / 2e-3 | CG4 chain epsilon/delta 0.08, 0.04, 0.02, 0.01, 0.005; CG6 at 0.005 (mesh check) |
| mesh_big | 4e-4 / 1.5e-3 | CG5, CG6, CG6 quadrature 48 at 0.005 (the source); about 3-4 CG6 nodes across the outer width |
| mesh_hdg_a/b/c | 1.5e-3/3e-3, 1e-3/2e-3, 7.5e-4/1.5e-3 | candidate HDG evolution meshes (`mesh.npz` for `--mesh-npz`) |

Solver: DOLFINx CG (`projects.diocotron.studies.equilibrium_catalog.run_contours`, hybridge-diocotron), PyPardiso
with 16 threads for the pilot and medium stages, MPI MUMPS with 16 ranks bound to cores 0-15 for the big mesh
(fallback: PyPardiso), `pde_tolerance` 1e-11, quadrature 40 (48 for the check), audit subdivisions 64.
Checks: `check_refinement` medium CG6 vs big CG6 (mesh), CG5 vs CG6 (degree), quadrature 40 vs 48.

## Running

`launch_source.py` (detached: `setsid nohup .../diocotron-fenicsx/bin/python -u launch_source.py >
driver_stdout.log 2>&1 < /dev/null &`), stages A-E with `STAGES_DONE`, final marker `SOURCE_DONE`, logs and
commands in `logs/`, inputs in `inputs/`, archive in `equilibria/`, stage directory for the HDG driver in
`stages/undercut_star/cg6/`, `selected_source.json`. Resuming skips every existing output.
Started 2026-10-21 09:37 UTC, pgid 2754424 (`kill -- -2754424`).

`inspect_source.py` (HYBRIDGE venv, `PYTHONPATH=.` from hybridge) draws `preview/inner_band_source.{pdf,png}`
and writes the table `preview/inner_band_source.md` once `SOURCE_DONE` exists.

## Results

First chain (2026-10-21 09:37-11:10 UTC, pgid 2754424): stages A and B accepted; the big-mesh CG5 solve converged
(dual residual 5e-15) on 16 MPI ranks and serially, and both were rejected by the lower-contour audit
(`LOWER_CONTOUR_AUDIT_UNRESOLVED`, logs/big_cg5_r0p005*_audit1e-10.log, SOURCE_FAILED_20261021T1110.json): audit
sub-lattice vertices fell within the value tolerance, 1e-10 times the torsion maximum (1.4e-11 in potential), of
the outer-edge level at every refinement but n = 5. `audit_tolerance_check.py` replays the audit on the medium CG6
state: its values differ from long-double evaluations by at most 5e-18 (audit_tolerance_check.json), so the
tolerance exceeds round-off by six orders and the rejections were chance proximity on 0.5M cells. The big stage
now uses `crossing_value_tolerance` 1e-12 (`BIG_AUDIT_TOLERANCE`; failed inputs in inputs/rejected_audit_1e-10/).

Second chain (2026-10-22 09:17-09:58 UTC, pgid 268099): big CG5, CG6 and CG6 q48 accepted on 16 MPI ranks
(202-248 s each; every audit resolved at n = 2 and 4 with no near-level vertices), checks and export done,
`SOURCE_DONE`. Selected source: big_cg6_q48_01 (archive id big_cg6_q48_01-9ce279b69af2).

Preview: `preview/inner_band_source.{png,pdf,md,json}` (table copied here).

| quantity | value |
|---|---|
| window m; delta; epsilon/delta | 0.00525 (= 7 delta); 0.00075; 0.005 |
| band d; L/d; T_c | 0.0377; 108; 151 |
| wall gap c/d: min; mean; max | 2.66; 3.65; 5.62 |
| transition width/d, outer; inner (mean) | 0.012; 0.228 |
| mass; phi_max; residual | 0.1576; 0.0056596; 6.4e-15 |
| source mesh triangles; CG6 dofs | 507,612; 9,139,897 |
| outer width in CG6 nodes, mean (min) | 6.5 (5.2) |
| mesh medium CG6 vs big CG6: phi, rho, grad phi | 1.53e-08, 6.95e-08, 5.69e-05 |
| degree CG5 vs CG6 (big mesh): phi, rho, grad phi | 5.40e-07, 1.86e-05, 4.79e-04 |
| quadrature 40 vs 48 (big mesh, CG6): phi, rho, grad phi | 9.24e-16, 9.80e-14, 5.16e-13 |

| mesh | triangles | outer/inner size | outer width in cells, mean (min) | gap cells near wall, median | interior cells, median | est. p=6 GPU GiB |
|---|---|---|---|---|---|---|
| mesh_big | 507,612 | 0.0004/0.0015 | 1.08 (0.87) | 0.0150 (0.40 d) | 0.0148 (0.39 d) | 51 |
| mesh_hdg_a | 116,450 | 0.0015/0.003 | 0.29 (0.22) | 0.0150 (0.40 d) | 0.0150 (0.40 d) | 12 |
| mesh_hdg_b | 193,064 | 0.001/0.002 | 0.43 (0.34) | 0.0150 (0.40 d) | 0.0150 (0.40 d) | 19 |
| mesh_hdg_c | 286,608 | 0.00075/0.0015 | 0.58 (0.42) | 0.0150 (0.40 d) | 0.0150 (0.40 d) | 29 |

Status: equilibrium shown to the user 2026-10-22; no evolution run until the user confirms it.

