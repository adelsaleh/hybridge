# Mid band: Section 7.3 repeated with the band about one thickness from the wall (2026-10-22)

Request (user, 2026-10-21): "repeat the whole experiment in section 7.3 but with a band having a few more units away
from the boundary so that the diameter of triangles outside the band and near the wall are not far from the
diamter of the triangles in the inner section. before running, compute the equilibrium and show it to me. it has
to be a very well resolved equilibrium to take a big mesh and high polnomial order, and scale mpi accordingly.
once I confirm the equilibrium, we can relaunch the campaign. (take a small espilon)"

First attempt: `e7_inner_band_20261021` (m = 7 delta, mean gap 3.65 d). Rejected by the user on 2026-10-22: "band is
too far from boundary, it does not follow the non-convex edges of the boundary ... the band should be closer to
the boundary but not as much as the first one was in the articles".

Choice: `e7_inner_band_20261021/position_scan/position_candidates.{png,json}` draws archived CG3 states
(epsilon/delta 0.08) of the predictor path of `e7_near_wall_equilibrium_20261020` between the Fig. 16 band and the
rejected one. The user chose nearwall_005 (archive id nearwall_005-29bf40587d94):

| quantity | nearwall_005 (CG3, epsilon/delta 0.08) |
|---|---|
| delta; m | 1.0184e-3; 2.2356e-3 = 2.195 delta |
| thickness d | 0.0396 |
| wall gap / d: min; mean; max | 0.59; 1.03; 1.70 |

The band follows all five undercuts and enters every arm (Fig. 16 band: mean gap 0.31 d).

## Chain

`launch_source.py` (detached, resumable; pgid in `launch.json`, stop `kill -- -<pgid>`): medium mesh graded around
the nearwall_005 level sets, CG4 epsilon/delta 0.08 -> 0.005, CG6; big mesh graded around the medium CG6 contours,
CG5, CG6, CG6 quadrature 48; checks (mesh, degree, quadrature); export (stage directory, candidate HDG meshes
a/b/c with grading 0.6, `SOURCE_DONE`). Every solve on 16 MPI ranks (MUMPS), serial PyPardiso fallback; contour-audit
tolerance 1e-12 (see the launcher docstring). `inspect_source.py` (HYBRIDGE venv) draws
`preview/mid_band_source.{png,pdf,md,json}`.

Started 2026-10-22 11:56 UTC, pgid 463841.

## Results

Attempts (launcher docstring has the details): the first big mesh (outer 4e-4, 1,001,738 triangles, 18.0M CG6
unknowns; `rejected_1M/`) exhausted the 125 GB host on 16 MUMPS ranks and swapped until the 3 h timeout
(12:43-18:02 UTC); the second (outer 5e-4, 745,106 triangles; `rejected_745k/`) hit the memory floor at CG6 both on
16 ranks and serially (during the second Newton refactorization). The launcher now has a memory guard (kill below
8 GiB available, counted as a failed attempt), waits for memory before each stage and forwards its stop signal to the
running stage. Final chain 2026-10-22 18:36-19:45 UTC (pgid 575273): big mesh 539,990 triangles (outer 5e-4, buffer
1.5e-3, grading 0.5), CG5 on 16 ranks (230 s, lowest available 53 GiB), CG6 and CG6 q48 serial (898 s each, lowest
available 34 GiB), checks, HDG meshes (grading 0.6), `SOURCE_DONE`. Selected source: big_cg6_q48_01.

Preview: `preview/mid_band_source.{png,pdf,md,json}`; quick looks of the medium states: `preview/snapshot_*.png`
(`snapshot.py <state>`).

| quantity | value |
|---|---|
| window m; delta; epsilon/delta | 0.0022356 (= 2.19518 delta); 0.00102; 0.005 |
| band d; L/d; T_c | 0.0439; 189; 264 |
| wall gap c/d: min; mean; max | 0.53; 0.90; 1.52 |
| transition width/d, outer; inner (mean) | 0.012; 0.229 |
| mass; phi_max; residual | 0.3753; 0.0027917; 2.2e-15 |
| interior density at the centre | 9.9e-05 |
| source mesh triangles; CG6 dofs | 539,990; 9,722,701 |
| outer width in CG6 nodes, mean (min) | 6.0 (4.5) |
| mesh medium CG6 vs big CG6: phi, rho, grad phi | 2.44e-07, 1.55e-06, 4.47e-04 |
| degree CG5 vs CG6 (big mesh): phi, rho, grad phi | 1.35e-06, 1.86e-05, 1.06e-03 |
| quadrature 40 vs 48 (big mesh, CG6): phi, rho, grad phi | 7.15e-16, 3.29e-14, 1.91e-13 |

| mesh | triangles | outer/inner size | outer width in cells, mean (min) | gap cells near wall, median | interior cells, median | est. p=6 GPU GiB |
|---|---|---|---|---|---|---|
| mesh_big | 539,990 | 0.0005/0.0015 | 1.00 (0.75) | 0.0087 (0.20 d) | 0.0150 (0.34 d) | 54 |
| mesh_hdg_a | 160,982 | 0.0015/0.003 | 0.33 (0.23) | 0.0085 (0.19 d) | 0.0150 (0.34 d) | 16 |
| mesh_hdg_b | 285,528 | 0.001/0.002 | 0.50 (0.35) | 0.0090 (0.20 d) | 0.0150 (0.34 d) | 29 |
| mesh_hdg_c | 442,242 | 0.00075/0.0015 | 0.67 (0.48) | 0.0086 (0.20 d) | 0.0150 (0.34 d) | 44 |

Status: equilibrium shown to the user 2026-10-22; no evolution run until the user confirms it.

