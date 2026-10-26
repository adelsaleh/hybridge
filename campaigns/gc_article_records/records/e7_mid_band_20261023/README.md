# Mid band moved outward: Section 7.3 repeat, band slightly farther from the wall (2026-10-23)

Request chain (user): 2026-10-21 "repeat the whole experiment in section 7.3 but with a band having a few more units
away from the boundary ... before running, compute the equilibrium and show it to me ... (take a small espilon)";
2026-10-22 the m = 7 delta band (`e7_inner_band_20261021`) was too far and no longer followed the undercuts; the user
chose nearwall_005 from `e7_inner_band_20261021/position_scan/`, computed in `e7_mid_band_20261022` (mean wall gap
0.90 d, minimum 0.53 d at epsilon/delta 0.005); 2026-10-23: "good, slightly move the band away from the boundary,
(a bit more)".

Choice: the next state outward on the archived predictor path of `e7_near_wall_equilibrium_20261020`, nearwall_003
(archive id nearwall_003-3416e67ebe05): delta 1.0720e-3, m = 2.5045 delta, d 0.0406, wall gap 0.71 / 1.22 / 1.98 d
(min / mean / max) at epsilon/delta 0.08 (CG3), against 0.59 / 1.03 / 1.70 d for nearwall_005 on the same path.

## Chain

`launch_source.py`, copied from `e7_mid_band_20261022` with only the window and start state changed (its docstring
records the memory limits found there): medium mesh graded around the nearwall_003 level sets, CG4 epsilon/delta
0.08 -> 0.005 and CG6 on 16 MPI ranks; big mesh (outer 5e-4, buffer 1.5e-3, grading 0.5), CG5 on 16 ranks, CG6 and
CG6 q48 serial; checks; HDG meshes a/b/c; `SOURCE_DONE`. Memory guard: an attempt is killed below 8 GiB available.
`inspect_source.py` (HYBRIDGE venv) draws `preview/mid_band_source.{png,pdf,md,json}`; `snapshot.py <state>` draws a
quick look of any accepted state.

Started 2026-10-23 08:47 UTC, pgid 10678 (`kill -- -10678`).

## Results

The medium CG6 solve (722,678 triangles, 13.0M unknowns) hit the memory floor on 16 ranks and serially
(09:14-09:35 UTC; `medium_cg6_rejected/`), so the medium stage ends at CG5 and the mesh check compares medium CG5
with big CG5 (launcher docstring). Final chain 09:35-10:43 UTC (pgid 23901): medium CG5 on 16 ranks (292 s), big mesh
528,564 triangles, big CG5 on 16 ranks (222 s), big CG6 and CG6 q48 serial (854 s each, lowest available 35 GiB),
checks, HDG meshes, `SOURCE_DONE`. Selected source: big_cg6_q48_01.

Preview: `preview/mid_band_source.{png,pdf,md,json}`; quick looks: `preview/snapshot_*.png`.

| quantity | value |
|---|---|
| window m; delta; epsilon/delta | 0.0026848 (= 2.50452 delta); 0.00107; 0.005 |
| band d; L/d; T_c | 0.0451; 180; 251 |
| wall gap c/d: min; mean; max | 0.64; 1.07; 1.77 |
| transition width/d, outer; inner (mean) | 0.012; 0.229 |
| mass; phi_max; residual | 0.3759; 0.0032698; 3.2e-15 |
| interior density at the centre | 1.1e-04 |
| source mesh triangles; CG6 dofs | 528,564; 9,517,033 |
| outer width in CG6 nodes, mean (min) | 6.2 (4.5) |
| mesh medium CG5 vs big CG5: phi, rho, grad phi | 4.85e-07, 4.09e-06, 8.23e-04 |
| degree CG5 vs CG6 (big mesh): phi, rho, grad phi | 1.30e-06, 1.90e-05, 1.05e-03 |
| quadrature 40 vs 48 (big mesh, CG6): phi, rho, grad phi | 7.07e-16, 3.58e-14, 2.23e-13 |

| mesh | triangles | outer/inner size | outer width in cells, mean (min) | gap cells near wall, median | interior cells, median | est. p=6 GPU GiB |
|---|---|---|---|---|---|---|
| mesh_big | 528,564 | 0.0005/0.0015 | 1.03 (0.74) | 0.0092 (0.20 d) | 0.0150 (0.33 d) | 53 |
| mesh_hdg_a | 158,484 | 0.0015/0.003 | 0.34 (0.26) | 0.0088 (0.20 d) | 0.0150 (0.33 d) | 16 |
| mesh_hdg_b | 279,662 | 0.001/0.002 | 0.52 (0.41) | 0.0087 (0.19 d) | 0.0150 (0.33 d) | 28 |
| mesh_hdg_c | 433,406 | 0.00075/0.0015 | 0.69 (0.52) | 0.0084 (0.19 d) | 0.0150 (0.33 d) | 43 |

Against `e7_mid_band_20261022` (same measurement): wall gap 0.64 / 1.07 / 1.77 d against 0.53 / 0.90 / 1.52 d.

Status: equilibrium shown to the user 2026-10-23; no evolution run until the user confirms it.

