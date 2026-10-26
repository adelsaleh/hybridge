# Band placement on the undercut star (2026-10-25)

## Question

On the non-convex star-like polygon (the undercut star: ten re-entrant corners of 244.6° and five of 216.4°), how do
the preservation and the long-time behaviour of a band equilibrium depend on how close the band lies to the
center? The quantities are the onset, the mode the band selects from round-off, its rate, and the rolled-up state.

User decisions:
- Seven bands from the wall to the center, all with resolved edges (ε/δ = 0.08).
- Every run starts from the HDG discrete equilibrium (`--equilibrium-transfer consistent`).
- The other geometries get one band and one run each (`../e7_transfer_20261025/launch_natural.py`).

## Bands

The archived states come from the predictor path of `../e7_near_wall_equilibrium_20261020` (CG3) and the inner
band of `../e7_inner_band_20261021` (CG4). The thickness d is nearly constant along the predictor path, which moves
m and δ together.

| label | source state | m/δ | δ | d | L/d | middle-contour clearance mean/d (min) | T_c | HDG mesh (triangles) |
|---|---|---|---|---|---|---|---|---|
| wall_031 | nearwall_018 | 1.00 | 7.5e-4 | 0.0342 | 263 | 0.62 (0.40) | 335 | 189,548 |
| wall_072 | nearwall_009 | 1.69 | 9.2e-4 | 0.0376 | 228 | 1.00 (0.68) | 289 | 196,962 |
| wall_122 | nearwall_003 | 2.50 | 1.07e-3 | 0.0406 | 200 | 1.46 (1.01) | 252 | 193,332 |
| wall_148 | nearwall_000 | 2.92 | 1.14e-3 | 0.0419 | 186 | 1.70 (1.19) | 236 | 188,646 |
| mid_25 | continuation_out pl_out_02 (CG4, 3e-3 mesh) | 5.0 | 7.5e-4 | 0.0338 | 160 | 2.93 (2.34) | | 135,848 |
| inner_410 | inner band medium CG4 | 7.0 | 7.5e-4 | 0.0337 | 120 | 4.38 (3.31) | 151 | 107,946 |
| center | continuation_inward pl_inward_02 (CG4, 5e-3 mesh) | 9.0 | 7.5e-4 | 0.0338 | | (4.47 min) | | 94,262 |

- **Labels** give the mean outer gap of the position scan, in units of d.
- **The CG source only needs to start Newton.** Each run's initial state is the discrete equilibrium of the HDG
  scheme, found by Newton from the projected CG state on the run's own mesh.
- **HDG meshes** come from `make_source_mesh.py --contours`: outer size 1.5e-3, inner 3e-3, bulk 0.025, corners 2.5e-3.

## New placements

- **`continue_midpoint.py` (killed).** Warm midpoint continuation with `run_contours` from the inner band (m = 7δ), CG4,
  ε/δ = 0.08, uniform 3e-3 mesh (665,984 triangles, 5.3M dofs), PyPardiso on four threads: two outward steps took
  6.7 h. `continuation_out/` holds m = 6δ (pl_out_01, clearance 3.36 d) and m = 5δ (pl_out_02, 2.68 d by the scan,
  2.93 d mean middle-contour clearance): mid_25 is pl_out_02 (`selected_new.json`).
- **`continue_inward.py` (detached, pgid 1546703).** The inward part on a uniform 5e-3 mesh (242,008 triangles) with
  PyPardiso on 16 threads, m = 8, 9, …, 16δ into `continuation_inward/`: about 320 s per 0.2δ step (six Newton
  iterations of 44 s each, of which 4 s are the factorization).
- **Center band: m = 9δ, not the fold.** At fixed δ the family folds (disk analogy: near m ≈ 9.5δ here). At the fold
  the linearization of the equilibrium problem is singular, so the consistent-transfer Newton would be
  ill-conditioned and a zero-frequency radial mode would mix into the run. The innermost band is therefore the 9δ
  state (pl_inward_02, `selected_new.json`); the continuation was stopped at 19:15 UTC once it existed.

## Runs

- **`launch_placement.py` (done for six bands).** p = 6, SI-BDF2, dt 0.05, T = 200, from the discrete equilibrium;
  band half-width factor 0.5 (at 1 the nearest-point map of wall_031 folds); no strip profiles. Records
  `placement_runs.jsonl`; marker `placement_partial.json` (center not prepared). 17 to 27 min per run alone.
- **`launch_extension.py a|b` (detached, lanes a pgid 1563259, b pgid 1563260).** Every placement again to T = 300 into
  `runs_t300/`, two runs side by side on the device (a wall-band run needs 18 GiB): lane a = wall_031, wall_122,
  mid_25, center; lane b = wall_072, wall_148, inner_410. About 0.6 s per step each when paired. Lanes a and b read
  `placements.json` before the center band existed, so lane a skips it (`extension_a_partial.json`); **lane c**
  (pgid 1589734) runs the center band once lane b has finished wall_148, so that at most a wall band, inner_410 and
  the center band share the device. Do not relaunch lane a after lane c has run (it would rerun the center band:
  each lane checks only its own records). Markers `EXTENSION_<LANE>_DONE`, `extension_<lane>_partial.json`; resume
  skips runs with an ok record. The article reads `runs_t300/` where a run has finished there, else `runs/`.
- **`analyze_placement.py`** → `placement_analysis.json`: corner share of the departure at t = 1, wall density near
  the corners, along-band localization in the linear growth. **`analyze_longtime.py`** → `longtime_analysis.json`:
  vortices (regions of ρ > 1/2 between 1 and 20 d²), unbroken share, their wall clearance and five-fold order at
  every saved state past the onset.

## Results at T = 300 (`runs_t300/`, analyze_onset definitions; all lanes done 22:39 UTC)

| band | seed max A(1) | mode (kd) | rate | onset t(1e-3) (predicted) | e_ρ(25) | largest vortex count (t) | count at 300 |
|---|---|---|---|---|---|---|---|
| wall_031 (0.62 d) | 1.2e-10 (m = 25) | 30 (0.72) | 0.072 | 207 (203) | 1.1e-5 | 15 (300) | 15 |
| wall_072 (1.00 d) | 5.6e-14 | 33 (0.91) | 0.134 | 184 (174) | 2.5e-8 | 32 (240) | 20 |
| wall_122 (1.46 d) | 5.1e-14 | 30 (0.94) | 0.153 | 160 (156) | 2.0e-8 | 29 (210) | 25 |
| wall_148 (1.70 d) | 3.7e-14 | 30 (1.01) | 0.166 | 159 (144) | 1.8e-8 | 27 (210) | 16 |
| mid_25 (2.93 d) | 6.7e-14 | 23 (0.90) | 0.156 | 156 (152) | 2.3e-8 | 22 (200) | 11 |
| inner_410 (4.38 d) | 9.0e-14 | 16 (0.83) | 0.174 | 151 (135) | 2.5e-8 | 17 (210) | 9 |
| center (5.91 d) | 7.2e-14 | 13 (0.80) | 0.172 | 146 (130) | 2.8e-8 | 14 (190) | 8 |

The T = 300 reruns reproduce the T = 200 runs exactly where the seed is above round-off (wall_031); from round-off
the onset of a rerun differs by up to two time units and the fitted rate by up to 0.013 (mid_25: 0.169 → 0.156), the
scatter between realizations. The article uses the T = 300 runs throughout.

- **Wall.** The rate falls within about 1.5 d of the wall, as for a sharp planar strip next to a wall at φ = 0
  (largest rate at edge gap c − d/2, in `../e7_transfer_20261025/make_figures_natural.py`), but less steeply.
  `analyze_placement.py` (→ `placement_analysis.json`): averaged over the saved states of the linear growth
  (dominant mode 1e-8 to 1e-4; a single state is not enough, the ratio flips with the phase of the travelling wave),
  the departure per unit mass is 1.27–1.35 in the fifth of the band farthest from the wall and 0.80–0.90 in the
  nearest for every band from 1.00 d to 2.93 d (1.14 / 0.85 at 4.38 d, flat at 0.62 d). Since the far bands show it
  too, it is not a wall effect and is not used to explain the rates.
- **Corners.** Only wall_031 is seeded by the corners: its density on the wall is about 1e-2, 2.5e-2 next to the
  corners (wall_072: 6e-6, wall_122: 2e-10); at t = 1, 83% of its departure lies within one thickness of the 15
  re-entrant corners (1.4% of the area; wall_072: 7%; `analyze_placement.py`); its spectrum is a comb at multiples of five.
- **Modes.** From round-off the fastest mode wins (kd 0.80–1.01 from 1.00 d inward); mode 30 for three of the four
  bands within 1.7 d, five-fold only for wall_031 by its spectrum.
- **Long time** (`analyze_longtime.py`, supplement S9). Each band rolls up into about as many vortices as its mode
  number (32/33, 29/30, 27/30, 22/23, 17/16, 14/13), then the vortices merge: by t = 300 most bands keep a half to two
  thirds of them (1.46 d: 0.86). Near the wall the vortices settle in the arms; the inner bands' vortices wander
  through the interior. wall_031 breaks last (t ≈ 300) into 15 vortices on the wall, three per arm (five-fold order
  0.99). The five-fold order of the others changes by up to 0.4 between saved states and spans up to 0.7 over a run:
  no lasting arrangement. Sampled minimum of ρ_h −0.15 to −0.52 at t = 300 (unlimited scheme, under-resolved
  filaments): late counts indicative only.

## Files

| Path | Content |
|---|---|
| `continue_midpoint.py`, `mesh_continuation/`, `continuation_{out,in}/`, `equilibria/` | new placements by continuation |
| `prepare_placements.py` → `placements.json`, `stages/undercut_star/<label>/`, `meshes/<label>/` | stages and HDG meshes |
| `launch_placement.py` → `runs/` | the controls to T = 200 |
| `continue_inward.py` → `mesh_inward/`, `continuation_inward/` | the inward continuation (center band) |
| `launch_extension.py` → `runs_t300/` | all placements to T = 300, lanes a, b and c |
| `analyze_placement.py`, `analyze_longtime.py` | corner seed and localization; vortices after the roll-up |
| `logs/` | mesh, continuation and driver logs |
