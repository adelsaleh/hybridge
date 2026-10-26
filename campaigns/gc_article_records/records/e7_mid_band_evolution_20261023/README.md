# Mid-band evolution: Section 7.3 repeated on the band one thickness from the wall (2026-10-23)

Request (user, 2026-10-23, after confirming the equilibrium of `e7_mid_band_20261023`): "you can rerun the
campaign now with the newer band. preservation tests are important, and I guess you can mention how the geometry
can affect the equilibrium simulation, ie, how does the wall 'naturally' generate the instability due to the lack
of resolution. from this, maybe we can find a more convincing perturbation test that actually is intrinsic to that
geometry and tell us a bit about the physics." Follow-up: "the wall still plays a role in the shape of the near
wall equilibrium. also, are we sure that a very difficult wall with a rough corner cannot generate an instability
if not well resolved? that's an interesting question to ask."

Plan: `~/.claude/plans/radiant-orbiting-harp.md` (approved 2026-10-23). Previous campaign on the 0.27 d band:
`e7_revision_20261020` (launchers, analysis and figure scripts adapted here).

## Source and meshes

`e7_mid_band_20261023`: CG6 source `big_cg6_q48_01-390b47b31cd8` (528,564 triangles), window m = 2.5 delta,
epsilon/delta = 0.005, d = 0.0451, L/d = 180, T_c = 251, wall gap 0.64 / 1.07 / 1.77 d (min / mean / max), outer
10-90 % width 0.012 d, inner 0.23 d. HDG meshes evaluating the same source: b (`mesh_hdg_b`, 279,662 triangles,
0.52 cells across the outer edge), c (`mesh_hdg_c`, 433,406, 0.69 cells; reference), and the corner variants of b
made here with the source study's `make_source_mesh.py` (`logs/mesh_b_corner_*.log`): `mesh_b_corner_none`
(corner size 0.015 = bulk, i.e. no corner refinement) and `mesh_b_corner_fine` (corner size 7.5e-4), identical to
b in every other size.

Geometry of the band (outer contour; `e7_mid_band_20261023/stages/undercut_star/cg6/contours.npz`): ten
narrowest-gap points at 0.64 d, two per arm neck (undercuts), among them the band point nearest (0.17, 0.82),
the site of the previous experiment; valleys between the arms 0.78 d; arm tips 1.76-1.77 d, e.g. (0.674, -0.928).
8 % of the band has a gap below 0.8 d, 33 % below 1.0 d, 92 % below 1.5 d.

Planar two-edge strip with the wall image (`make_figures.planar_rate` of `e7_revision_20261020`): gap 0.64 d,
maximum 0.146 at kd 0.735, unstable to kd 1.12; 1.07 d, 0.178 at 0.79; 1.77 d, 0.195 at 0.805; no wall 0.201 at
0.795; previous band 0.27 d, 0.071 at 0.52.

## Seeds

Packet (`launch_evolution.py`): carrier kd 0.75, three wavelengths (25 d long), transverse half-width d, crest at
the undercut (0.17, 0.82) or at the widest gap (0.674, -0.928), wall cutoff 0.04 d, spline grid d/48;
outer-edge displacement 0.002 d (1/6 of the outer width; scan 0.004 and 0.001 on mesh c). Whole-band sinuous modes
l = 21 (kd 0.73) and l = 9 (kd 0.31) in the flow angle, crest at the undercut, linearized only. Host checks of
every seed: `seed_checks/` (`run_seed_checks.sh`).

## Lanes (`launch_evolution.py --lane ...`, chained by `launch_chain.py`)

| lane | mesh | runs |
|---|---|---|
| smoke | c | seed, p = 6, 20 steps: peak device memory |
| b | b | controls p = 6, 5 (T = 200, dt 0.025) and p = 4 (dt 0.05); seed 0.002 and linear at p = 6, 5 (T = 100); control / seed / linear at dt 0.05 and control / seed at dt 0.0125 (p = 6, T = 100; linear dt 0.05 to 120) |
| c | c | controls p = 6, 5, 4; seeds 0.002 (movie), 0.004, 0.001 and linear at p = 6; seed 0.002 and linear at p = 5 |
| g | b | linearized, p = 6, dt 0.05, T = 120: packet kd 0.75 at the widest gap, packets kd 0.3 and 0.5 at the undercut, modes l = 21 and 9 |
| k | bk0, bk1 | controls p = 6, dt 0.025, T = 200 on the corner variants (against ctrl_b_p6_dt0p025) |

SI-BDF2, compatible drift, Poisson atol 5e-14; states every 5 to t = 100 then every 10; strip profiles every
0.5 (1024 angles, 32 transverse points, half-width 0.9 d); diagnostics every 10 steps. Accuracy campaign: no
GPU lock, waits for 30,000 MiB (b meshes) or 45,000 MiB (c) free before each run; all runs sequential on the
device. Records `evolution_<lane>_runs.jsonl`, markers `EVOLUTION_<LANE>_DONE`, chain marker `CHAIN_DONE`,
pgid and stop command in `chain_launch.json`.

## Status

Chain started 2026-10-23 13:09 UTC, process group 103064 (`kill -- -103064`), `driver_stdout.log`, lane logs
`evolution_<lane>_stdout.log`, driver logs `runs/<lane>/logs/`. Seed checks (13:06-13:09 UTC): area change
<= 2.5e-9, no point outside the wall, edge displacement ratio 1.000 (mode 21: 1.003), outer-edge displacement
9.0e-5 = 0.002 d, inner-edge 1.5e-5 (packets) / 2.1e-5 (modes). Smoke run (mesh c seed, p = 6, 20 steps):
ok in 160 s, peak 39.6 GiB (device total 40.8 GiB). Lane b started 13:12 UTC: the p = 6 control runs at 0.40 s
per step (8000 steps, about 55 min).

### First control (mesh b, p = 6, dt 0.025, T = 200; 13:12-14:22 UTC, 4170 s, 0.43 s per step, peak 23.5 GiB)

`analyze_preservation.py` and `analyze_geometry.py` ran on it (`preservation_summary.json`, `geometry_summary.json`,
`geometry_envelopes.npz`, `figures/e7_midband_preservation`, `figures/e7_midband_geometry`, `tables/e7_midband_preservation.tex`).

| quantity | value |
|---|---|
| e_rho at t = 1; 5; 10; 25; 50 | 1.2e-3; 2.0e-3; 2.5e-3; 3.6e-3; 6.5e-2 |
| e_phi at t = 1; 25; 50 | 7.3e-7; 3.5e-5; 1.7e-3 |
| e_E at t = 5; 25; 40; 50 | 4.3e-3; 4.6e-3; 5.0e-3; 8.9e-3 |
| rho_h extrema, t <= 25; whole run | -0.032 / 1.032; -1.34 / 1.80 |
| outer edge shift (rms / max, units of d) at t = 1; 10; 25 | 4.8e-4 / 1.5e-3; 9.4e-4 / 2.3e-3; 1.2e-3 / 5.4e-3 |
| mass drift | 4.9e-13 |
| largest band mode of phi_h - phi_h(0): t = 0; 1; reaches 1e-5; 1e-4; 1e-3; 1e-2 | 1.9e-17; 1e-6; t = 15.3; 28.3; 42.3; 58.8 |
| dominant modes (kd) and rates over t = 15-42 | 25 (0.87) 0.174; 20 (0.70) 0.153; 30 (1.05) 0.183 |
| planar rate at kd 0.87: gap 0.64 d; 1.07 d; 1.77 d | 0.137; 0.176; 0.193 |
| along-band envelope max/min at t = 1; 5; 40; 60 | 11; about 4; about 5; about 2 |
| envelope-gap correlation at t = 1; 5; 20; 40; 60 | -0.64; -0.55; -0.40; +0.04; +0.60 |

Reading: the projection of the 0.52-cell outer edge seeds a band-mode amplitude of 1e-6 within the first time
unit, and the mode then grows at the strip-like rate (0.17 against the planar 0.178 at the mean gap) until the
roll-up (e_rho reaches 1 at t = 100, the band is fully rolled up at t/T_c = 0.3). The early envelope is five-fold
periodic and anticorrelated with the gap; by t = 40 the five-fold modulation is a factor 5 (local rate
differences of 0.04, the planar range between 0.64 d and 1.77 d is 0.05) and has drifted by 0.16 L, so it no
longer lines up with the gap. Whether the 1e-6 seed drops with mesh c and with p: lane c and the p = 5 controls.

### Seed and linearized runs on mesh b (p = 6, dt 0.025, T = 100; 2188 s and 2012 s, peaks 27.0 and 21.9 GiB)

`analyze_evolution.py` (`evolution_summary.json`) and `analyze_linear.py` (`linear_summary.json`) ran on them.

| quantity | t = 5 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|
| control drift / response, potential | 3.2 % | 3.8 % | 4.3 % | 5.7 % | 5.1 % | 9.6 % |
| control drift / response, density | 67 % | 47 % | 17 % | 9 % | 12 % | 41 % |
| nonlinear - linearized per unit amplitude, potential | 0.2 % | 0.3 % | 1.1 % | 3.4 % | 11 % | 88 % |
| nonlinear - linearized per unit amplitude, density | 17 % | 27 % | 59 % | 89 % | 97 % | 100 % |
| mean envelope of the response / initial (seed; linearized) | 1.7; 1.7 | 3.7; 3.7 | 14.3; 14.3 | 62.3; 62.3 | 317; 319 | 2.6e3; 1.1e4 |

Linearized packet (kd 0.75, crest at the undercut, gap 0.64 d): potential rate 0.142 over t = 0-10, 0.141 over
10-40, 0.164 over 40-100 (planar: 0.146 at the crest gap, 0.178 at the mean gap); the dominant kd of the strip
spectrum moves from 0.77 (t = 0) to 0.87-0.91 (t = 10-20) and back to 0.80-0.84 (t = 40-60).

Reading: in potential the matched-control subtraction is clean (drift 3-6 % of the response to t = 50) and the
nonlinear response follows the linearized one to 1 % at t = 20 and 3 % at t = 30; the roll-up begins near
t = 40 (displacement 0.002 d x e^(0.15 x 40) = 0.8 d). In density the comparison with the linearized run fails
from the start (13 % at t = 0, 59 % at t = 20): a 0.002 d displacement of a 0.012 d edge is not a small
perturbation of the density profile, and the control drift in density is the projection relaxation of the
0.52-cell edge (e_rho 2e-3 to 3e-3, static), which the subtraction removes. The density comparisons are
therefore reported as what they are (sharp-edge limits); the potential is the physical comparison. Main
amplitude stays at 0.002 (the ratios in potential do not improve with 0.004 enough to matter; the amplitude
scan on mesh c quantifies the nonlinear part).

Static band geometry on the profile grid (`geometry_static_b.npz`, `analyze_geometry.py`): the middle contour
phi = m lies 0.35 d (0.31-0.39) from the sharp outer edge and 0.65 d (0.61-0.70) from the diffuse inner one;
local thickness 0.93-1.10 d; outer-edge gap 0.64 / 1.07 / 1.77 d (min / mean / max, as for the source); outer
width 0.0098-0.0135 d = 0.39-0.66 cells on mesh b; curvature kappa d from -0.40 to +0.50; drift speed
0.028-0.036 (= 0.62-0.81 d per time unit). The planar prediction along the drift uses the measured edge gap
and the local kd of the advected packet (0.75-0.97 for the kd 0.75 packet). Linearized undercut packet:
measured mean-envelope amplification 3.7 / 14 / 62 / 319 at t = 10 / 20 / 30 / 40 against 5.8 / 35 / 194 / 1087
for the planar rate integrated along the drift (path average 0.174) and 4.3 / 19 / 80 / 344 at the constant
crest-gap rate 0.146: the packet starts at the undercut rate and keeps a deficit of 0.03 in rate for about
30 time units (a sinuous seed projects onto the growing and the damped edge-wave pair, cosh rather than exp),
then grows at 0.164 against the path average 0.174.

### p = 5 control on mesh b (dt 0.025, T = 200; 3188 s)

| quantity | p = 6 | p = 5 |
|---|---|---|
| e_rho at t = 1; 25; 50 | 1.2e-3; 3.6e-3; 6.5e-2 | 2.3e-3; 6.3e-3; 7.5e-2 |
| e_E at t = 25 | 4.6e-3 | 9.4e-3 |
| edge shift max (d), t <= 25 | 5.4e-3 | 6.7e-3 |
| min rho_h, t <= 25 | -0.032 | -0.062 |
| largest band mode reaches 1e-3 at | t = 42 | t = 41 |
| dominant kd; rate | 0.87; 0.174 | 0.87; 0.176 |
| along-band envelope, p5 / p6 at t = 0.5; 5; 10; 20-60 | 2.5; 1.8; 1.45; 1.21 | |
| correlation of log envelopes p5 vs p6 at t = 0.5; 5; 20 | 0.83; 0.89; 1.00 | |

Reading: the degree halves the projection error (e_rho, e_E, min rho_h) and the first departure along the band is
2.5 times larger at p = 5, with the same five-fold pattern (correlation 0.83-0.89); from t = 20 on the two
envelopes differ by a constant factor 1.21 with identical patterns and identical growth (0.176 against 0.174),
so the degree sets the seed of the natural instability and the physics sets its growth; the onset time moves
by only one time unit because the seed of the growing mode differs by 1.2, not by 2.5 (the early departure is
mostly the static relaxation of the projection, which projects weakly on the growing mode).

### Lane b complete (12 runs ok, 13:12-20:44 UTC; `EVOLUTION_B_DONE`)

Controls on mesh b (`preservation_summary.json`):

| control | e_rho(1) | e_rho(25) | e_E(25) | min rho_h (t <= 25) | A_max = 1e-3 at | kd; rate |
|---|---|---|---|---|---|---|
| p = 4, dt 0.05 | 4.6e-3 | 1.1e-2 | 1.9e-2 | | t = 39.5 | 0.87; 0.178 |
| p = 5, dt 0.025 | 2.3e-3 | 6.3e-3 | 9.4e-3 | -0.062 | 41.0 | 0.87; 0.176 |
| p = 6, dt 0.025 | 1.17e-3 | 3.57e-3 | 4.6e-3 | -0.032 | 42.2 | 0.87; 0.174 |
| p = 6, dt 0.05 | 1.17e-3 | 3.57e-3 | 4.6e-3 | | 42.5 | 0.87; 0.174 |
| p = 6, dt 0.0125 | 1.17e-3 | 3.58e-3 | 4.6e-3 | | 42.1 | 0.87; 0.174 |

The time step does not enter the control (e_rho identical to three digits for dt 0.05 / 0.025 / 0.0125); the
degree halves e_rho(1) and e_E(25) per degree and moves the onset of the natural instability by 1.3-1.5 time
units per degree (seed of the growing mode x1.3 per degree, growth rate unchanged).

Packet, response-normalized refinement differences in percent, density / potential (`evolution_summary.json`):

| pair | t = 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|
| p = 5 vs 6 (mesh b) | 13.2 / 0.09 | 10.8 / 0.06 | 4.9 / 0.03 | 2.9 / 0.12 | 4.0 / 0.36 | 11.1 / 1.0 | 52.9 / 8.5 | 72.5 / 25.4 |
| dt 0.05 vs 0.025 | 0.60 / 0.19 | 0.45 / 0.25 | 0.90 / 0.38 | 4.8 / 0.47 | 8.0 / 0.45 | 13.6 / 0.57 | 20.4 / 1.31 | 36.6 / 3.09 |
| dt 0.025 vs 0.0125 | 0.18 / 0.05 | 0.12 / 0.06 | 0.23 / 0.09 | 2.4 / 0.12 | 5.3 / 0.15 | 9.1 / 0.28 | 13.6 / 0.47 | 27.3 / 1.43 |

Potential: second order in time through t = 40 (ratios 3.8-4.2), then 2.0-2.8 as the roll-up starts; the degree
changes the potential response by 0.1 % or less during the linear phase (t = 10-40). Density: the sharp edge
keeps the degree difference at 3-13 % and the time-step pairs leave second order after t = 30.

Linearized packet (kd 0.75, undercut): rates identical to four digits for p = 5 and p = 6 at dt 0.025 and for
dt 0.05 (0.142 / 0.141 / 0.164 over 0-10 / 10-40 / 40-100; 0.157 over 100-120); nonlinear - linearized potential
0.34 % (p = 6) / 0.55 % (p = 5) at t = 0, 1.06 % at t = 20 for every resolution.

### Mesh c control (p = 6, dt 0.025, T = 200; 6512 s, 0.67 s per step, peak see records) and the five-fold seed

| quantity | mesh b | mesh c |
|---|---|---|
| e_rho at t = 1; 25; 50 | 1.17e-3; 3.6e-3; 6.5e-2 | 3.6e-4; 1.9e-3; 6.4e-2 |
| e_phi at t = 1; 25 | 7.3e-7; 3.5e-5 | 3.0e-7; 3.5e-5 |
| e_E at t = 25; 50 | 4.6e-3; 8.9e-3 | 1.3e-3; 7.5e-3 |
| rho_h extrema, t <= 25 | -0.032 / 1.032 | -0.004 / 1.004 |
| outer edge shift rms at t = 1; 25 (d) | 4.8e-4; 1.2e-3 | 1.8e-4; 1.1e-3 |
| HDG potential vs CG6 source at t = 0 | 3.6e-6 | 3.5e-6 |
| largest band mode reaches 1e-5; 1e-4; 1e-3 at | 15.3; 28.2; 42.2 | 15.5; 28.5; 42.2 |
| dominant modes (kd); rates | 25 (0.87) 0.174; 20 (0.70) 0.153; 30 (1.05) 0.183 | identical |
| along-band envelope, b / c at t = 0.5; 2; 5; 10; >= 20 | 3.8; 2.0; 1.3; 1.07; 1.01-1.02 (correlation 1.00) | |

Band-mode spectrum of phi_h(t) - phi_h(0) (`band_phi_mode_amplitudes`, modes 1-80), every control (b p4/p5/p6, c p6):
only the multiples of 5 carry amplitude. At t = 1 the top modes are 20, 30, 40, 50, 10 (4.3e-7, 3.7e-7, 3.0e-7,
1.9e-7, 1.8e-7 on b p6; 4.3e-7, 3.6e-7, 3.0e-7, 1.8e-7, 1.7e-7 on c p6; 5.3e-7 ... on b p5; 6.9e-7 ... on b p4),
growing linearly in t from t = 0 (2.2e-7 at t = 0.5, 8.2e-7 at t = 2); the mean amplitude of the other modes
is 2e-11 to 1e-8 at all times (ratio 4,000-17,000). By t = 20 the fastest of the five-fold harmonics lead: 25
(kd 0.87), 20 (0.70), 15 (0.52), 30 (1.05). So the natural instability is seeded by a five-fold, geometry-shaped
departure of the initial state from stationarity whose amplitude is the same on meshes b and c and changes by
only 1.23 per degree, not by the projection noise of the sharp edge (which differs by 3.2 between the meshes and
never rises above the round-off floor in any non-five-fold mode). Candidate: the CG6 source's own discretization
error. The HDG potential of the projected density differs from the CG6 potential by 3.5e-6 on both meshes
(the CG5-CG6 source difference is 1.3e-6), rho_h = W(phi_CG6) is stationary for the drift of phi_CG6, not for
the more accurate HDG drift, and u_h . grad rho_h = W' grad^perp(phi_h - phi_CG6) . grad phi_CG6 is then the
source error, smooth and five-fold. Decisive test (not run; needs the user's go-ahead and a CG5 stage export):
a control on mesh b, p = 6, T = 60 from the CG5 source: the seed should grow by the CG5/CG6 error ratio and the
onset move earlier by ln(ratio)/0.175. The corner variants (lane k) test the corner resolution separately.

### Lane c complete (9 runs ok, 20:44-05:24 UTC; `EVOLUTION_C_DONE`; mesh c seed peak 41.6 GiB, control 36.6, linear 33.8)

Controls on mesh c (p = 4 at dt 0.05, p = 5 / 6 at dt 0.025):

| control | e_rho(1) | e_rho(25) | e_E(25) | A_max = 1e-3 at | kd; rate |
|---|---|---|---|---|---|
| mesh c, p = 4 | 2.2e-3 | 6.7e-3 | 8.1e-3 | t = 39.5 | 0.87; 0.178 |
| mesh c, p = 5 | 8.9e-4 | 3.3e-3 | 3.3e-3 | 41.0 | 0.87; 0.176 |
| mesh c, p = 6 | 3.6e-4 | 1.9e-3 | 1.3e-3 | 42.2 | 0.87; 0.174 |

The onset times are the same as on mesh b for every degree (39.5 / 41.0 / 42.2): the seed of the growing mode
depends on p, not on h, while e_rho(1) drops by 2.1-2.6 per degree and by 2.1-3.2 from mesh b to c. Linearized
packet on mesh c: rates 0.1424 / 0.1412 / 0.1637 (0-10 / 10-40 / 40-100) for p = 5 and 6, identical to mesh b.

Roll-up figure (`make_figures.py rollup:30,40,50,60:c`, mesh c, p = 6, dt 0.025, alpha 0.002): the packet
bulges after the undercut at t = 40, rolls up in the following lobe by t = 50 and is a vortex with a filament at
t = 60; sampled density extrema per frame: t = 30 [-0.003, 1.003], 40 [-0.13, 1.14], 50 [-0.21, 1.24],
60 [-0.28, 1.21].

Packet on the reference mesh c (p = 6, dt 0.025; `evolution_summary.json`), percent:

| quantity | t = 0 | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| mesh b vs c, rho / phi | 1.7 / 0.56 | 10.5 / 0.04 | 7.1 / 0.03 | 3.3 / 0.01 | 1.6 / 0.08 | 1.6 / 0.23 | 3.4 / 0.48 | 6.0 / 0.61 | 13.1 / 1.5 |
| p = 5 vs 6 on c, rho / phi | 0.9 / 0.38 | 8.0 / 0.04 | 5.6 / 0.03 | 2.7 / 0.01 | 2.3 / 0.10 | 3.8 / 0.33 | 11.0 / 1.05 | 52 / 8.4 | 72 / 25 |
| nonlinear - linearized, phi, alpha 0.004 / 0.002 / 0.001 | 0.43 / 0.22 / 0.11 | 0.52 / 0.26 / 0.13 | 2.13 / 1.06 / 0.52 | 6.9 / 3.4 / 1.7 | 22.6 / 11.3 / 5.6 | 70 / 46 / 25 | 95 / 88 / 76 | | |
| nonlinear - linearized, rho, alpha 0.004 / 0.002 / 0.001 | 25.6 / 13.2 / 6.6 | 45.7 / 25.1 / 13.9 | 81 / 60 / 35 | 95 / 89 / 76 | 99 / 97 / 94 | | | | |
| control drift / response, rho / phi | 0 | 17.0 / 3.6 | 7.2 / 4.2 | 6.6 / 5.7 | 11.7 / 5.1 | 18.6 / 4.2 | 41 / 9.5 | | |

The potential departure from the linearized response is proportional to alpha at every time through t = 40
(ratios 2 : 1 : 1/2), so it is the leading nonlinear correction and the linearized run is the alpha -> 0 limit;
the potential response per unit amplitude of the three seeds and of the linearized run agree to four digits
through t = 20 (0.2731). The density departure is proportional to alpha as well but starts at 13 % (alpha 0.002):
the finite displacement of the 0.012 d edge. Mesh b vs c and p = 5 vs 6 change the potential response by 0.1 %
or less during the linear phase and by 0.5 / 1 % at t = 60 (roll-up), the density response by 2-10 % early
(edge representation) and 3-11 % at t = 60. Main amplitude stays 0.002.

### Lane g complete (5 linearized runs ok, 05:24-07:27 UTC; mesh b, p = 6, dt 0.05, T = 120; `EVOLUTION_G_DONE`)

| seed | rate 0-10 | 10-40 | 40-100 | 100-120 | final modes (kd) | pattern speed / drift |
|---|---|---|---|---|---|---|
| kd 0.75 packet, undercut (0.64 d) | 0.142 | 0.141 | 0.164 | 0.157 | 24, 25, 23 (0.84, 0.87, 0.80) | 0.82 |
| kd 0.75 packet, arm tip (1.72 d) | 0.098 | 0.176 | 0.146 | 0.175 | 23, 22, 21 (0.80, 0.77, 0.73) | 0.63 |
| kd 0.5 packet, undercut | 0.132 | 0.124 | 0.165 | 0.141 | 24, 23, 25 | 0.65 |
| kd 0.3 packet, undercut | 0.112 | 0.112 | 0.147 | 0.161 | 25, 24, 26 (0.87, 0.84, 0.91) | 0.36 |
| whole-band mode 21 (kd 0.73) | 0.165 | 0.163 | 0.160 | 0.154 | 21, 26, 16 | 0.58 |
| whole-band mode 9 (kd 0.31) | 0.096 | 0.109 | 0.152 | 0.166 | 24, 19, 29 | 0.70 |

Planar strip, mean gap: 0.178 at kd 0.75-0.79, 0.177 at 0.73, 0.176 at 0.87, 0.149 at 0.5, 0.097 at 0.3; crest
gap 0.64 d: 0.146 (0.75), 0.126 (0.5), 0.081 (0.3); tip gap 1.72 d: 0.193. Free strip 0.201.

Readings.
* Five-fold mode coupling: the wall's symmetry couples flow-angle mode l to l +- 5n. Mode 21 ends in 21, 26, 16;
  mode 9 in 24, 19, 29 (= 9 + 15, 9 + 10, 9 + 20), and the natural instability, seeded at multiples of 5, stays
  there. A single-mode seed ends in the fastest member of its family, kd 0.84-0.87, the band's preferred wave.
* The wave pattern travels at 0.36-0.82 of the middle-contour drift (slower for longer waves): the band is a
  strongly sheared flow (the outer edge drifts at about 0.048, the inner edge near 0.004, the middle contour at
  0.032), and the coupled edge waves ride between the edge speeds. The planar predictions along the drift now
  use the measured pattern speed.
* Whole-band mode 21: rate 0.163 at all times (planar mean 0.175, free 0.20; the 0.27 d band gave 0.056-0.061
  against 0.071). Its envelope develops a five-fold standing pattern with maxima at the arm tips and minima at
  the necks (max/min 1.6 at t = 5, 3.4 at t = 20, 3.8 at t = 40, 1.7 at t = 60, 4.6 at t = 100: a beat of the
  coupled family), the same pattern as the natural instability of the controls. The pattern does not travel
  with the wave, and the planar rate integrated along the measured wave path predicts a drifting pattern of
  opposite phase (correlation -0.9 at t = 20-60): the growing disturbance is a global mode whose envelope is
  tied to the wall's shape, not an advected packet accumulating local growth.
* Packets: the undercut and the tip packets reach the same regime (0.15-0.165, kd 0.8-0.87) after different
  transients; their amplifications differ by 1.6 at t = 40 and 1.25 at t = 60 (tip ahead), far less than
  the planar path integral predicts, which also mis-assigns which one leads.

### Lane k complete; chain complete (`CHAIN_DONE` 2026-10-24 09:43 UTC; 28 runs, all ok at the first attempt, 20.6 h)

Corner variants of mesh b (p = 6, dt 0.025, T = 200; 4163 s and 4178 s, peaks 23.3 and 23.8 GiB):

| control | corner cells | e_rho(1) | e_rho(25) | e_E(25) | e_E(50) | A_max = 1e-3 at | envelope / mesh b (t >= 20) |
|---|---|---|---|---|---|---|---|
| mesh b, no corner refinement (bk0) | bulk size 0.015 = 0.33 d | 1.16e-3 | 6.6e-3 | 4.6e-3 | 3.2e-2 | t = 33.2 | 4.2 (pattern correlation 0.999) |
| mesh b | (see config) | 1.17e-3 | 3.6e-3 | 4.6e-3 | 8.9e-3 | 42.2 | 1 |
| mesh b, fine corners (bk1) | 7.5e-4 = 0.017 d | 1.19e-3 | 3.5e-3 | 4.7e-3 | 6.9e-3 | 45.2 | 0.65 (1.000) |

Reading: the corner resolution does not touch the projection of the edge (e_rho(1) and e_E(25) identical) but
sets the seed of the natural instability: without corner refinement the five-fold seed is 4.2 times larger and
the instability reaches 1e-3 nine time units earlier; with corner cells at 7.5e-4 it is 0.65 times mesh b's and
the onset moves three time units later, always with the same pattern and the same growth. Together with the mesh
b / c identity (same corner size, same seed) and the weak degree dependence (x1.23 per degree, consistent with a
corner singularity limiting the convergence in p): the seed of the natural instability is the Poisson error at
the re-entrant corners of the wall (the undercuts), which is smooth along the band, five-fold, and converges with
the corner cell size, not with the band's edge resolution. An under-resolved corner therefore does generate the
instability, through the potential's corner singularity, not through the sharp edge.

This supersedes the CG5-source test proposed above: the seed follows the HDG-vs-CG6 potential difference, which
the corner variants change by 4.1 and 0.69 while the source is fixed; a CG5-source control would now measure only
the residual source part.

## Final status (2026-10-24)

Campaign complete: 28 runs (smoke 1, b 12, c 9, g 5, k 2), all ok at the first attempt, 13:09 UTC 2026-10-23 to
09:43 UTC 2026-10-24. Analysis outputs: `preservation_summary.json`, `evolution_summary.json`,
`linear_summary.json`, `geometry_summary.json`, `geometry_envelopes.npz`, `geometry_static_{b,c,bk0,bk1}.npz`;
figures `figures/e7_midband_{setup,preservation,response,geometry,rollup}.{pdf,png}`; tables
`tables/e7_midband_{setup,preservation,refinement,rates}.tex` (regeneration commands in `figures/README.md`).
Article: Section 7.3 replaced (`article_drafts/section_7_3_midband.tex` is the inserted text,
`article_drafts/section_7_3_previous_0p27d.tex` the replaced one); the 0.27 d band is supplement Section S6
(`article_drafts/supplement_nearwall_0p27d.tex`); `PLAN.md` E7 entry updated. Driver logs under `runs/<lane>/logs/`
carry local time (UTC+2); the records and this README use UTC.

Full-domain frames (user request 2026-10-24): `make_figures.py frames[:label[:mesh[:n]]]` writes one PNG per
snapshot time to `runs/<lane>/frames/<run>/t<time>.png` (rho_h on a shared signed scale; for a seeded run also
the difference from the matched control on a per-frame symmetric scale), rendered from the `*_snapshots.npz`
fields on a 900-pixel-wide grid. Written for the reference seed (`runs/c/frames/mb_a0p002_c_..._seed`, 21
frames, t = 0-100) and its control (`runs/c/frames/mb_ctrl_c_..._control`, 31 frames, t = 0-200). The driver
itself saves no images; the movie of the reference seed is in `runs/c/movies/`.

Where the natural instability starts (user request 2026-10-24): `make_figures.py where` writes
`figures/e7_midband_where` (controls on mesh b without corner refinement, mesh c, mesh b with fine corners; t = 1,
5, 15, 30, 45) and `make_figures.py where-frames:<mesh>` writes 60 frames (t = 1-60) to
`runs/<lane>/frames/<control>_where/`, for c, bk0 and bk1. Each draws the probe strip in place, coloured by the
along-band envelope of phi_h(t) - phi_h(0) (Gauss-weighted transverse RMS, smoothed over one wavelength of band
mode 25) relative to its maximum at that time; re-entrant corners marked (10 arm-neck corners turning 65 deg, 5
valley corners turning 36 deg). The pointwise maximum is not used: early on it sits on the sharp outer edge at
isolated angles (0.1-0.2 % of the strip points), the projection noise of the edge.

Envelope near the corners (within 1.5 d of the middle contour) over the rest of the band:

| control | t = 0.5 | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|
| no corner refinement, necks / valleys | 1.60 / 1.39 | 1.96 / 1.12 | 2.10 / 0.80 | 1.78 / 0.41 | 1.37 / 1.00 | 0.49 / 2.14 | 1.27 / 0.53 |
| mesh c | 1.74 / 1.17 | 2.09 / 0.95 | 2.20 / 0.64 | 1.82 / 0.32 | 1.37 / 0.98 | 0.47 / 2.16 | 1.27 / 0.54 |
| fine corners | 1.16 / 1.78 | 1.20 / 1.70 | 1.27 / 1.61 | 1.34 / 1.27 | 1.26 / 1.17 | 0.49 / 2.13 | 1.27 / 0.54 |

The departure first appears where the band passes the arm-neck corners (the undercuts, narrowest gap 0.64 d):
twice the band average there during the first five time units, weakest at the arm tips. With fine corner cells
the concentration at the necks drops to 1.2-1.3. From t = 10 on all three controls carry the same pattern, that of
the growing global mode, which beats between necks and valleys (0.47-0.49 / 2.1 at t = 20).

Control snapshot gallery (user request 2026-10-24): `make_figures.py control-gallery[:mesh]` writes
`figures/e7_midband_control` (mesh c, p = 6, dt 0.025: rho_h at t = 0, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150,
200, one signed density scale). Sampled extrema: unchanged [-0.003, 1.003] through t = 50, [-0.02, 1.03] at 60,
[-0.16, 1.15] at 70, [-0.22, 1.22] at 80 (rolled up into about 25 vortices), [-0.48, 1.49] at 200 (vortices merged
into the arms). No movie exists for the controls (the campaign wrote one for the reference seed only, to t = 100).
