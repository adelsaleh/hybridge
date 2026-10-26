# Geometry-tied lump perturbations on every domain (E7 search, 2026-10-20)

Request (user): "put all these preliminary results in the article and pursue the search for meaningful
perturbations on each of our geometries".

## Design

On the star, the varicose mode-5 seed with crests at the lobe tips (e7_five_sites_20261019, `var5tipsA045p6`)
gathers each sector's band into one short thick lump; each lump rolls up into one vortex, so the band breaks
into exactly five vortices, one per lobe, which last to T = 200. The rule behind it: a lump rolls up into a
single vortex when it is shorter than the fastest wavelength of its own thickness (about 8 thicknesses), which
holds for sectors of about 25 d with the edges moved apart by 0.45 d (necks about 0.1 d; at 0.25 d the necks
form their own vortices). The search carries this over to the other domains: the mode gives sectors of
20–31 d and the phase puts crests on the band's sharpest turns.

Curvature extrema of the middle contour (normalized arclength s, `features.py` here (run from hybridge with `PYTHONPATH=.`),
`analyze_band_evolution.contour_curvature`, smoothing d/4):

| Domain | L/d | convex turns (s; point) | concave bends |
|---|---|---|---|
| star | 127 | 0.186, 0.386, 0.586, 0.786, 0.987 (lobe tips) | 0.086, 0.286, ... (valleys) |
| Ugly Pac-Man | 127 | 0.001 (-0.426, 0.629) upper jaw; 0.382 (-0.226, -0.693) lower jaw | 0.213 (0.426, -0.102) mouth bend |
| ITER | 139 | 0.306 (1.361, 1.905) lower corner (X-point side); 0.875 (1.198, 8.804) upper corner | — |
| horseshoe | 154 | 0.074 (-0.642, -0.487), 0.474 (0.642, -0.487) hairpin turns | 0.267 inner-arc top |

- Ugly Pac-Man: mode 5 with a crest on the upper jaw turn puts crests at s = 0.0, 0.2, 0.4, 0.6, 0.8, within
  0.01–0.02 of the mouth bend and the lower jaw turn.
- ITER: the corners are 0.569 and 0.431 of the band apart (4:3), so mode 7 puts crests on both (mismatch 0.003).
- Horseshoe: the turns are exactly 0.4 apart, so mode 5 puts crests on both turns, on the top of the inner arc and
  on two points of the outer arc symmetric about its top (mirror-symmetric pattern).

## Runs (round 1, exploration resolution)

Embedded CG6 meshes, p = 4, compatible drift, SI-BDF2, dt = 0.05, T = 100, Poisson atol 5e-14 / rtol 5e-11,
varicose seeds (`--varicose`), edge displacement 0.45 d unless stated. `launcher.py`, two lanes sharing the GPU
(no lock), outputs `runs/lp<name>_*`.

| Run | Domain | Mode | Crest at | Edges/d |
|---|---|---|---|---|
| `star5valleys` | star | 5 | valleys (lumps between the lobes) | 0.45 |
| `pacman5` | Ugly Pac-Man | 5 | upper jaw turn | 0.45 |
| `pacman5mouth` | Ugly Pac-Man | 5 | mouth bend | 0.45 |
| `iter7` | ITER | 7 | upper corner (and lower corner) | 0.45 |
| `iter5` | ITER | 5 | lower corner, X-point side | 0.45 |
| `horseshoe5` | horseshoe | 5 | hairpin turns | 0.45 |
| `horseshoe5A055` | horseshoe | 5 | hairpin turns | 0.55 |

Seed checks (`seed_check/`, host): area change of the displaced contours ≤ 3e-6, no point outside the wall,
measured/target edge displacement 0.79 (star valleys: the total thickening equals that of the lobe-tip seed),
1.05 (Pac-Man), 0.90 (ITER), 0.99 (horseshoe).

Started 2026-10-20 11:48: lane a pgid 681682 (`kill -- -681682`), lane b pgid 681744 (`kill -- -681744`);
markers `LUMPS_DONE_a`, `LUMPS_DONE_b`. About 0.4 s per step with two runs sharing the GPU, 17 min per run;
0.6–1.4 s per step and 25–50 min per run once lane c's p = 6 run joined.

`horseshoe5A055` ran out of device memory at t = 21 (13:09, AMGX "out of memory"; three runs on the device:
p = 6 Pac-Man about 23 GB, two p = 4 runs about 11 GB each; `runs.jsonl` status `oom`, log
`runs/logs/horseshoe5A055.log`), which ended lane b. The launcher now reruns runs whose record is not `ok`
(`Campaign.finished` counts failed attempts as finished) and gives this run `MIN_FREE` 24 GB. Lane b was
restarted at 13:12 behind a wait for `LUMPS_DONE_a` (so that at most two runs share the device), then again at
13:43 with `MIN_FREE` 20 GB once lane a was done (a p = 6 run alone leaves about 23 GB free): pgid 910690
(`kill -- -910690`); attempt 2 of `horseshoe5A055` started 13:43 next to `pacman5p6`.

## Round 2: article resolution (lane c)

`pacman5p6`, `iter7p6`, `horseshoe5p6`: the seeds of `pacman5`, `iter7`, `horseshoe5` at p = 6, T = 200 (the E7
resolution, as `var5tipsA045p6` on the star). Started 12:01, pgid 707563 (`kill -- -707563`), marker
`LUMPS_DONE_c`; each run waits for 30 GB free device memory, so the lane starts once one p = 4 lane is done
(about 23 GB per p = 6 run, 70–100 min each). A file `SKIP_<name>` here makes the lane skip that run.

## Results

Coherent vortices (cores of {ρ_h > 1/2} eroded by 0.4 d in regions holding ≥ 2 % of the band area; saved states
every 2.5 up to t = 60, then every 5; `runs/*_coherent.json`, `analysis_<name>/` with tracks and spectra):

| Run | N at t = 0 | N = ℓ for t ∈ | max N | N(T = 100) | what happens |
|---|---|---|---|---|---|
| `pacman5` | 5 | 0–30 | 9 | 6 | five lumps → five vortices by t = 20; each vortex passing the mouth bend (s = 0.21) or the lower jaw turn (s = 0.38) is strained and sheds a satellite (7–9 from t = 33, 6 from t = 75) |
| `iter7` | 7 | 47–100 (and 0–40) | 8 | 7 | seven lumps → seven vortices that drift at the equilibrium speed (one sector per 25 time units) through both corners without change; one brief satellite at t = 42–45 |
| `star5valleys` | 5 | 0–10 | 10 | 10 | each valley lump (band within 3 d of the wall, concave contour, measured thickening 0.79 of the target) rolls up into two vortices at once: 10 from t = 13, orbiting in pairs within their sectors from t ≈ 60; none settles in a lobe (compare `var5tipsA045`: lumps at the lobe tips give exactly five) |
| `iter5` | 5 | 0–28 | 7 | 6 | five lumps (28 d sectors) → five vortices by t = 20; the longer necks roll up near the corners into one (t = 30) then two (t = 38–52) secondary vortices; 6 at T = 100 |
| `pacman5mouth` | 5 | 0–28 | 9 | 7 | same as `pacman5` with the lumps shifted by ≤ 0.03 of the band: five vortices by t = 20, satellites from t = 30 (6–9), 7 at T = 100 |
| `horseshoe5A055` | 5 | 15–42 | 8 | 5 | thicker lumps (0.55 d): same as `horseshoe5` (satellites 6–8 while passing the turns, 5 at T = 100) |
| `horseshoe5` | 5 | 20–42 | 8 | 5 | five lumps → five vortices by t = 20 (6–7 briefly during roll-up); satellites while vortices pass the hairpin turns (8 at t = 45–55, 6–7 to t = 70); the two vortices nearest the turns move more slowly along the band; 5 at T = 100 |

The curvature of the band relative to its thickness orders the three: κd ≈ 0.2 at the ITER corners (gentle),
0.32–0.41 at the Pac-Man jaw turns and −0.33 at the mouth bend, 0.25 at the star lobe tips, 0.40 at the horseshoe
turns. Three runs at one resolution and one amplitude do not establish a criterion.
Figures `figures/e7_uglypacman_lumps.*`, `figures/e7_iter_lumps.*`, `figures/e7_horseshoe_lumps.*`; the 25–28 min
runs peaked at 9.8 / 11.5 / 11.2 GB (0.6 s per step with three runs on the GPU).

**Placement in the article (12:42):** while this study ran, the article's Section 7 was reduced to the four
preservation controls and one planned localized perturbation (`run_outputs/e7_single_perturbation_20261020/`);
all perturbed runs, the search table and the lump figure now live in `supplementary/equilibria.tex`
(section "Lump seeds on the other domains", `fig:e7-lumps` with panels a–c, `tab:e7-search`), which reads frozen
copies under `figures/e7_exploratory_20261020/` and `tables/e7_exploratory_20261020/` (listed in that study's
`revision_manifest.json`). Build: `latexmk -pdf -outdir=supplementary supplementary/equilibria.tex` from the
article root. New rows and panels are copied there by hand after postprocessing.

Round 1 complete (`LUMPS_DONE_a` 13:41, `LUMPS_DONE_b` 14:08). Summary: the lump construction gives one vortex
per lump on every domain within the first twenty time units; what follows depends on the wall. Sectors longer
than about 25 d (ITER mode 5) leave necks that roll up into secondary vortices; sharp turns (κd ≳ 0.3, Pac-Man
jaws and mouth, horseshoe hairpins) strain passing vortices into satellites, and a vortex can be held at the
Pac-Man's lower jaw tip; the ITER corners (κd ≈ 0.2) leave the seven vortices unchanged; lumps placed where the
band hugs the wall (star valleys) split in two at once. Phase shifts of a few hundredths of the band and a
thicker lump (0.55 d) do not change the outcome.

Lane c (p = 6, T = 200): `pacman5p6`, `iter7p6`, `horseshoe5p6` — fold in with `./postprocess.py <name>`
(figure stems `e7_<g>_lumps_p6`), then switch the supplement's panels by hand.

| Run | N at t = 0 | N = ℓ for t ∈ | max N | N(T = 200) | what happens |
|---|---|---|---|---|---|
| `pacman5p6` (119 min, peak 19.7 GB) | 5 | 165–200 (and 0–30) | 9 | 5 | as `pacman5` to t = 100 (satellites 7–9 from t = 33, 6 from t = 60); 6 to t ≈ 160, then the satellites merge back and exactly five remain from t = 165 to T = 200 |
| `iter7p6` (48 min alone on the GPU, peak 22.5 GB) | 7 | 47–155 (and 0–40) | 8 | 6 | as `iter7`: seven vortices through both corners; two merge at t ≈ 160 and again at the end (6 at t = 160, 195, 200) |
| `horseshoe5p6` (41 min alone on the GPU) | 5 | 95–190 (and 20–42) | 8 | 4 | as `horseshoe5` to t = 100 (satellites while passing the turns), five from t = 95 to 190, two merge at the end (4 at t = 195, 200) |

Lane c complete (`LUMPS_DONE_c` 15:41). The p = 6 runs repeat their p = 4 sequences to t = 100 on all three
domains; the supplement's `fig:e7-lumps` now shows the p = 6 runs (panels a–c) and `tab:e7-search` has 19 rows.
