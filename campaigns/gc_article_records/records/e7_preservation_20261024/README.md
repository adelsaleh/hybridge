# Preservation controls for the rebuilt Section 7.2 (2026-10-24)

User decisions (2026-10-24): Section 7.3 (the localized displacement) is removed from the article; Section 7.2 is
rebuilt around what the code preserves, what ends preservation (the natural instability, seeded by the
discretization error at the wall's re-entrant corners and growing at the physical rate) and what the geometry
does; new runs P, F, K2, the scheme/time-step smoke lane T, and a stable thick band (lane S) after its equilibrium
is shown; Section 7 at most 6 pages; figures of the instability to t = 80-100; fastest recorded device presets
(the driver's `fast_profiles.fastest_device_overrides`: raw-CUDA assembly, face BSR, Legendre-modal trace,
native fb-hp-mg-pcg with cached Schur-Cholesky factors, `fast` policy at p = 6); verbose logging.
Plan: `~/.claude/plans/silly-strolling-valiant.md`.

## Why dt = 0.05 for the controls

The mid-band campaign ran the same control at dt 0.05, 0.025 and 0.0125 (mesh b, p = 6): e_rho identical to three
digits, onset of the natural instability (largest band mode at 1e-3) at t = 42.5 / 42.2 / 42.1, same modes and
rates. The time-step error contributes to the seed at the 7 % level at dt 0.05 (the 0.3 time-unit shift), far
below the corner seed; halving the step doubles the cost for nothing measurable. The controls here therefore
use dt 0.05 (lane K2 keeps 0.025 to compare directly with the existing corner variants), T = 100 (the roll-up
is complete by t = 80 on every band seen so far), and lane T decides the step of the long stable run.

## Meshes (`meshes/`, logs in `logs/`)

| mesh | from | sizes | triangles |
|---|---|---|---|
| mid_mesh_c_corner_none | `e7_mid_band_20261023/make_source_mesh.py` | outer 7.5e-4, inner 1.5e-3, corner 0.015 (= bulk) | 429,830 |
| mid_mesh_b_corner_1p5e-3 | same | outer 1e-3, inner 2e-3, corner 1.5e-3 | 281,286 |
| far_mesh_b_corner_none | `e7_inner_band_20261021/make_source_mesh.py` | outer 1e-3, inner 2e-3, grading 0.3, corner 0.015 | 189,214 |
| uglypacman_corner_none | `prepare_refined` (hybridge-diocotron), as `equilibrium_dt080_20261016/inputs/cg6_refined` but corner size = bulk 0.006 | band 0.0025, buffer 0.003, grading 0.3 | 209,996 (215,246 with corners 1e-5) |
| iter_corner_none | same, bulk 0.02, corner 0.02 | band 0.0075, buffer 0.009 | 240,476 (250,997 with corners 3e-5); `iter_corner_none_bulk0p018` was a first attempt with bulk 0.018 (277,664), unused |

`msh_to_npz.py` converts the `prepare_refined` meshes to `mesh.npz` for the driver's `--mesh-npz`.

## Lanes (`launch_controls.py --lane T|P|F|K2`, chained by `launch_chain.py`; `progress.sh` prints the state)

| lane | runs | source | notes |
|---|---|---|---|
| T | si-bdf2 and si-euler at dt 0.05 / 0.1 / 0.2 / 0.4, mesh b, p = 6, T = 40, diagnostics every step | mid band | stability of the long runs; about 15 min |
| P | Pac-Man and ITER: p = 6 on the corner-refined source mesh and on the same mesh without corner refinement, p = 5 and 4 on the source mesh; dt 0.05, T = 100 | `equilibrium_dt080_20261016/cg6_refined` (eps/delta 0.08) | the corner seed on two more domains; the existing uniform-mesh controls (`e7_revision_20261020/controls_resolved`) are the third resolution |
| F | mesh b p = 6 / 5 / 4, mesh c p = 6, mesh b without corner refinement p = 6; dt 0.05, T = 100 | `e7_inner_band_20261021` (gap 3.65 d) | the reach of the geometry |
| K2 | mesh c without corner refinement, mesh b corner 1.5e-3; p = 6, dt 0.025, T = 100 | mid band | corner ladder |

Common: unperturbed controls, SI-BDF2 (T: both schemes), diagnostics every 2 steps (band-mode amplitudes, e_rho,
e_phi, mass, extrema), strip profiles every 0.5 (1024 x 32, half-width 0.9), states at t = 1, 5, 10, 20, ..., 100,
driver log line every 10 steps in `runs/<lane>/logs/<run>.log`, records `controls_<lane>_runs.jsonl` (status, wall
time, peak memory, timings), markers `CONTROLS_<LANE>_DONE`, `CHAIN_DONE`.

## Status

Chain started 2026-10-24 20:46 UTC, process group 933803 (`kill -- -933803`), order T -> P -> F -> K2, about
14 h sequential on the device (T 15 min, P 6 h, F 3.5 h, K2 2 h at the measured 0.4-0.7 s per step).
`driver_stdout.log`, lane logs `controls_<lane>_stdout.log` (start/end/timings per run and the driver's last
lines), driver logs `runs/<lane>/logs/<run>.log`, `./progress.sh`.

### Lane T done (8 runs ok, 20:47-21:36 UTC; mesh b, p = 6, T = 40; 0.43 s per step for SI-BDF2, 0.50 s for SI-Euler, plus about 3 min of set-up per run)

| scheme, dt | largest band mode at t = 20; 40 | e_rho(40) | min rho_h(40) | energy drift (t = 40) |
|---|---|---|---|---|
| SI-BDF2 0.05 | 2.26e-5; 7.14e-4 | 1.67e-2 | -0.013 | 9e-11 |
| SI-BDF2 0.1 | 2.24e-5; 7.09e-4 | 1.67e-2 | -0.013 | |
| SI-BDF2 0.2 | 2.17e-5; 6.91e-4 | 1.63e-2 | -0.013 | |
| SI-BDF2 0.4 | 1.94e-5; 6.21e-4 | 1.50e-2 | -0.013 | |
| SI-Euler 0.05 | 2.05e-5; 5.82e-4 | 1.40e-2 | -0.013 | 6e-8 |
| SI-Euler 0.1 | 1.96e-5; 4.81e-4 | 1.20e-2 | -0.013 | 9e-8 |
| SI-Euler 0.2 | 1.82e-5; 3.81e-4 | 9.5e-3 | -0.013 | 1e-7 |
| SI-Euler 0.4 | 1.65e-5; 3.30e-4 | 7.0e-3 | -0.013 | 1e-7 |

Both schemes are stable at every step (no growth beyond the physical rate, bounded energy drift, mass drift
1e-14). SI-BDF2 keeps the instability's amplitude within 3 % of the dt 0.05 result through dt 0.2 and within
13 % at 0.4; SI-Euler damps it (first-order dissipation: 18 % low at dt 0.05, 54 % at 0.4) and needs 2.5 times
more transport iterations per step. Decision for the long stable run: SI-BDF2 at dt 0.2, with the half-step
check at 0.1 over the first circulation period.

### Onset law on the existing controls (`analyze_onset.py`, `onset_summary.json`; 21:40 UTC)

Seed = largest band mode of phi_h(t) - phi_h(0) at t = 5; onset = time at which it reaches 1e-3; predicted shift
= ln(seed_ref / seed) / rate_ref against the family's p = 6 reference; measured = difference of onsets.

| control | e_phi(1) | seed(5) | five-fold / others | onset | kd; rate | measured shift | predicted |
|---|---|---|---|---|---|---|---|
| mid band, mesh b p = 4 (dt 0.05) | 4.8e-6 | 2.8e-6 | 4500 | 39.5 | 0.87; 0.178 | -2.7 | -2.7 |
| mid band, mesh b p = 5 | 1.8e-6 | 2.1e-6 | 5300 | 41.0 | 0.87; 0.176 | -1.2 | -1.1 |
| mid band, mesh b p = 6 (reference) | 7.3e-7 | 1.8e-6 | 6000 | 42.2 | 0.87; 0.174 | 0 | 0 |
| mid band, mesh c p = 6 | 3.0e-7 | 1.7e-6 | 9600 | 42.2 | 0.87; 0.174 | 0.0 | +0.1 |
| mid band, mesh b no corner refinement | 1.3e-6 | 7.5e-6 | 97 | 33.2 | 0.87; 0.186 | -9.0 | -8.3 |
| mid band, mesh b fine corners | 7.0e-7 | 1.1e-6 | 7400 | 45.2 | 0.87; 0.170 | +3.0 | +2.5 |
| ITER resolved, uniform mesh, p = 4 / 5 / 6 | 3.2e-7 / 6.4e-8 / 1.5e-8 | 4.1e-8 / 3.2e-8 / 2.8e-8 | 1 | 71.5 / 73.0 / 73.7 | 0.65; 0.172 | -2.2 / -0.7 / 0 | -2.3 / -0.9 / 0 |
| Pac-Man resolved, uniform mesh, p = 4 / 5 / 6 | 9.7e-7 / 6.3e-7 / 5.5e-7 | 2.7e-6 / 2.3e-6 / 2.1e-6 | 1 | 46.0 / 47.2 / 48.0 | 0.71; 0.137 | -2.0 / -0.7 / 0 | -2.1 / -0.8 / 0 |
| plain star resolved, p = 4 / 5 / 6 | 4.6e-7 / 8.2e-8 / 1.5e-8 | 5.6e-10 / 4.1e-10 / 3.3e-10 | 7 / 35 / 41 | 99 / > 100 / > 100 | 0.66; 0.166 | | |
| horseshoe resolved, p = 4 / 5 / 6 | 6.6e-7 / 1.2e-7 / 2.1e-8 | 3.7e-11 / 2.6e-11 / 2.0e-11 | 1 | 98 / > 100 / > 100 | 0.73; 0.18 | | |
| sharp sources on corner-refined meshes (star, Pac-Man, ITER, horseshoe), p = 6 | 1e-5 | 1.6e-9 / 5.1e-9 / 3.0e-9 / 2.1e-9 | 1 | 80 / 74 / 77 / 73 | 0.74-0.86; 0.18-0.20 | | |

Eight pairs agree within 0.7 time units. Domains with re-entrant corners on uniform meshes (Pac-Man, ITER) carry
seeds of 1e-6 to 1e-8 that barely change with p (x1.3 per degree, like the mid band); domains without them (plain
star, horseshoe) sit two to four orders lower and reach 1e-3 only around t = 100 or later; the sharp sources on
the corner-refined meshes (corner cells 1e-5) have seeds of 2e-9 to 5e-9 on all four domains, Pac-Man and ITER
included. The horseshoe has no symmetry and no five-fold selection; the plain star's seed is five-fold at p = 6.

### Initial residual at the corners (`analyze_initial_residual.py`, `initial_residual.json`; host, 21:45-22:05 UTC)

t = 0 state of the mid-band controls, volume quadrature: transport residual r_h = grad^perp phi_h . grad rho_h,
its part due to the HDG potential's mismatch r_mis = grad^perp (phi_h - phi_CG) . grad rho_h, and the gradient
mismatch |grad (phi_h - phi_CG)|, integrated per cell and grouped by the distance of the cell to the nearest
re-entrant corner. The band has 12 % of its area within 1.5 d of a re-entrant corner.

| control | share of |r_h|^2 within 1.5 d | share of |r_mis|^2 | share of |grad dphi|^2 | grad-mismatch density within 1.5 d / beyond 3 d |
|---|---|---|---|---|
| mesh b, p = 6 | 0.51 | 0.46 | 0.45 | 1.8e-7 / 8.3e-9 (21x) |
| mesh c, p = 6 | 0.49 | 0.45 | 0.64 | 3.6e-8 / 7.3e-10 (49x) |
| mesh b, no corner refinement | 0.50 | 0.48 | 0.60 | 3.3e-7 / 8.1e-9 (40x) |
| mesh b, fine corners | 0.53 | 0.50 | 0.46 | 1.8e-7 / 8.2e-9 (23x) |
| mesh b, p = 5 | 0.47 | 0.43 | 0.40 | 6.1e-7 / 3.6e-8 (17x) |
| mesh b, p = 4 | 0.44 | 0.41 | 0.36 | 2.2e-6 / 1.6e-7 (14x) |

Half of the initial transport residual sits on the 12 % of the band nearest the re-entrant corners; the HDG-vs-
source gradient mismatch is 14-49 times denser there than away from the corners, its near-corner density doubles
without corner refinement (3.3e-7 against 1.8e-7) while the far density is unchanged (8.1e-9), and the fine
corners do not reduce it on this measure (the L2 norm is dominated by the edge projection; the five-fold seed is
the smooth component). The potential mismatch itself (dphi norm 1.7e-8 on b and c, 7.2e-8 on bk0, 1.2e-8 on bk1)
follows the seed ratios (4.2 and 0.69).

### Lane P done (8 runs ok, 21:36-23:50 UTC; Pac-Man 0.29-0.37 s per step at p = 6, ITER 0.4 s)

Corner-refined CG6 sources (eps/delta 0.08, corner cells 1e-5 at the Pac-Man mouth and the ITER corners) evolved on
their own mesh at p = 4 / 5 / 6 and on the same mesh without corner refinement (k0) at p = 6, dt 0.05, T = 100,
against the existing uniform-mesh controls of the uniform-mesh sources (`e7_revision_20261020/controls_resolved`):

| control | e_phi(1) | seed(5) | onset t(1e-3) | kd; rate | measured shift | predicted |
|---|---|---|---|---|---|---|
| Pac-Man, uniform mesh and source, p = 6 | 5.5e-7 | 2.1e-6 | 48.0 | 0.71; 0.137 | | |
| Pac-Man, corner-refined, p = 4 / 5 / 6 | 9.3e-9 / 1.5e-9 / 1.1e-9 | 5.8e-9 / 4.7e-9 / 4.0e-9 | 81.7 / 82.9 / 83.8 | 0.80; 0.177 | -2.1 / -0.9 / 0 | -2.1 / -0.9 / 0 |
| Pac-Man, corner-refined source on the mesh without corner refinement, p = 6 | 1.5e-7 | 5.7e-7 | 55.6 | 0.75; 0.132 | -28.2 | -28.1 |
| ITER, uniform mesh and source, p = 6 | 1.5e-8 | 2.8e-8 | 73.7 | 0.65; 0.172 | | |
| ITER, corner-refined, p = 4 / 5 / 6 | 2.5e-9 / 2.6e-10 / 1.4e-10 | 9.9e-10 / 7.3e-10 / 6.1e-10 | 94.5 / 96.2 / 97.2 | 0.77; 0.166 | -2.7 / -1.0 / 0 | -2.9 / -1.1 / 0 |
| ITER, corner-refined source on the mesh without corner refinement, p = 6 | 1.9e-9 | 9.3e-9 | 80.0 | 0.65; 0.171 | -17.2 | -16.3 |

Refining the corners of the mouth and of the ITER wall lowers the seed by 500 (Pac-Man) and 45 (ITER) and moves
the onset from t = 48 to 84 and from 74 to 97; removing the corner refinement from the HDG mesh alone, with the
same source, brings the seed back by 140 and 15 and the onset forward by 28 and 17 time units, as the law
predicts (28.1 and 16.3). With refined corners the Pac-Man seed converges with p like the other domains.

### Lane F done (5 runs ok, 23:50-01:00 UTC; band at 3.65 d: d 0.0377, L/d 108, T_c 151; 0.4 s per step on mesh b, 0.6 on c)

| control | e_rho(1) | e_phi(1) | seed(5) | five-fold / others | onset t(1e-3) | mode (kd); rate | measured shift | predicted |
|---|---|---|---|---|---|---|---|---|
| mesh b, p = 6 (reference) | 2.0e-3 | 5.9e-7 | 1.4e-7 | 530 | 62.1 | 10 (0.58); 0.162 | 0 | 0 |
| mesh c, p = 6 | 7.6e-4 | 1.4e-7 | 1.4e-7 | 340 | 62.1 | 10 (0.58); 0.162 | 0.0 | 0.0 |
| mesh b, p = 5 / 4 | 3.6e-3 / 6.5e-3 | 1.4e-6 / 3.5e-6 | 1.7e-7 / 2.2e-7 | 540 / 490 | 60.8 / 58.8 | 10; 0.163 / 0.164 | -1.3 / -3.3 | -1.2 / -3.0 |
| mesh b, no corner refinement, p = 6 | 2.1e-3 | 6.1e-7 | 7.1e-7 | 94 | 51.1 | 10; 0.165 | -11.0 | -10.1 |
| mid band, mesh b, p = 6, for comparison | 1.2e-3 | 7.3e-7 | 1.8e-6 | 6000 | 42.2 | 25 (0.87); 0.174 | | |

At 3.65 d from the wall the seed is 13 times smaller than at 1.07 d (the corner error decays with the distance),
identical on meshes b and c, x1.2 per degree and x5 without corner refinement (the law: -11.0 measured, -10.1
predicted); it is still five-fold (ratio 340-540). The band selects mode 10 (kd 0.58; L/d 108) instead of 25
(kd 0.87; L/d 180): the fastest member of the five-fold family near the strip's maximum. The rate, 0.162, is not
higher than at 1.07 d (0.174) although the wall image is negligible at 3.65 d (e^-2kc = 0.015): the equilibrium's
own profile (diffuse inner edge, 0.23 d), not the image, keeps both bands below the free strip's 0.20.

### Lane K2 done; chain complete (`CHAIN_DONE` 2026-10-25 02:50 UTC; 23 runs, all ok at the first attempt, 6.1 h)

| control (mid band, p = 6, dt 0.025) | corner cells | seed(5) | onset | rate |
|---|---|---|---|---|
| mesh b, no corner refinement | 0.33 d | 7.5e-6 | 33.2 | 0.186 |
| mesh c, no corner refinement | 0.33 d | 7.6e-6 | 33.1 | 0.186 |
| mesh b | 0.067 d | 1.8e-6 | 42.2 | 0.174 |
| mesh c | 0.067 d | 1.7e-6 | 42.2 | 0.174 |
| mesh b, corner cells 1.5e-3 | 0.033 d | 1.3e-6 | 44.0 | 0.172 |
| mesh b, fine corners | 0.017 d | 1.1e-6 | 45.2 | 0.170 |

The corner ladder: the seed depends on the corner cells alone (identical on meshes b and c at equal corner size,
with or without refinement) and converges slowly with them (7.5e-6, 1.8e-6, 1.3e-6, 1.1e-6 for 0.33, 0.067,
0.033, 0.017 d), as a corner singularity does; the onset follows ln(seed)/0.174.

### The onset from the initial state alone (2026-10-25, `analyze_onset.py`: `initial_rate_top`, `onset_from_initial_rate`)

User question: how much error is a user allowed, and can it be read off the initial state? The band-mode
departure grows linearly during the first time unit, A_m(t) ~ Adot_m t (two to forty steps), and a constant forcing
of an unstable mode gives A ~ (Adot/gamma) e^(gamma t), so the onset predicted from the initial state is
t* = ln(A* gamma / Adot_m) / gamma with Adot_m the initial rate of the mode that ends dominant and gamma its rate.
Against the measured onsets (A* = 1e-3):

| family | Adot_m (first time unit) | predicted t* | measured |
|---|---|---|---|
| mid band b / c, p = 6 | 1.4e-7 | 41 | 42.2 |
| mid band, no corner refinement (b, c) | 6.1e-7 | 31 | 33.2 / 33.1 |
| mid band, fine corners | 9.0e-8 | 44 | 45.2 |
| mid band p = 4 / 5 | 2.2e-7 / 1.7e-7 | 38 / 40 | 39.5 / 41.0 |
| Pac-Man uniform / corner-refined / refined source on mesh without corners | 2.2e-7 / 3.4e-10 / 5.4e-8 | 47 / 75 / 59 | 48.0 / 83.8 / 55.6 |
| ITER uniform / corner-refined / mesh without corners | 1.8e-9 / 2.6e-11 / 6.1e-10 | 67 / 94 / 73 | 73.7 / 97.2 / 80.0 |
| far band b / no corner refinement | 2.9e-8 / 1.5e-7 | 53 / 42 | 62.1 / 51.1 |
| sharp star / Pac-Man / ITER / horseshoe | 1.8e-10 / 3.8e-10 / 2.5e-10 / 3.7e-10 | 74 / 73 / 72 / 67 | 80.4 / 74.0 / 76.7 / 72.5 |
| plain star / horseshoe resolved, p = 6 | 2.1e-11 / 1.0e-13 | 96 / 117 | > 100 / > 100 |

The initial rate places the onset within 1-3 time units where the seed is large and within 5-9 where it is
small (the forcing is spread over several modes and the dominant one takes a few units to emerge); the amplitude
read at t = 5 places it within 1. So the precision requirement can be stated on the initial state: a user who
needs the band held for T_p must keep the initial band-mode departure rate below A* gamma e^(-gamma T_p), each
decade of Adot buying ln(10)/gamma = 13 time units (0.05 T_c) on the mid band; standard L2 measures (edge
projection error, semilinear residual after projection: 2.5e-4 on b, 2.0e-4 on c for identical seeds) do not
predict it.

## After the consistent transfer (2026-10-25)

These controls start from the projected transfer, and their seed is the transfer's corner mismatch. They are now in
supplement S8 of the article. Section 7.2 uses the consistent transfer (`../e7_transfer_20261025/`).
- `analyze_onset.py` has the family `consistent` (`../e7_transfer_20261025/runs/ct_*`). It is excluded from the
  within-family shift law.
- `make_figures_72.py prediction` draws `fig:e7-onset-prediction`: onset against the prediction from the initial rate,
  for both transfers.
