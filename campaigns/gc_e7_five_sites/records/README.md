# Five-fold seeds of the star band (E7 side study, 2026-10-19)

Question (user): "I once perturbed the star at 5 places and got exactly 5 coherent modes. Do you think you can
find the perturbation that does this? ... most likely the parts of the band closest to the convex corners."

The star, its equilibrium band and a seed placed at five symmetric places share the five-fold symmetry, which
the guiding-center dynamics keep up to the mesh: structures form and merge in fives. The band is closest to
the wall in the five valleys (clearance 0.130) and farthest at the five convex corners (lobe tips, 0.296).

## Runs

Exploration resolution: embedded CG6 star mesh (264,268 triangles), p = 4, compatible drift (postprocessed),
SI-BDF2, dt = 0.05, T = 200, Poisson atol 5e-14; band edges displaced by at most 0.05 d by the area-preserving
map of E7. `launcher.py` (detached, process group in `driver.json`, marker `X5_DONE`), outputs in `runs/`.

| Run id | Seed |
|---|---|
| `mode5tips` | χ ∝ sin(5θ + ψ), crests at the band points nearest the lobe tips |
| `bulgetips` | five outward bulges (θ-width 0.35) at the band points nearest the lobe tips |
| `bulgevalleys` | five outward bulges at the band points nearest the valleys |

Seed checks (host, `seed_check/`): area change ≤ 1.3e-8, edge displacement 0.97–1.02 of the target, no point
leaves the wall.

## Result: `mode5tips` gives five coherent vortices

Coherent vortex = connected component of {ρ_h > 0.5} holding at least 2 % of the initial band area
(`analyze_band_evolution.COHERENT_FRACTION`).

| t | structure | coherent vortices |
|---|---|---|
| 30 | five bulges, one per lobe | 1 (band) |
| 40–80 | each bulge rolls up into three vortices | 15 (10–15 by area) |
| 90–150 | pairs orbiting in each sector | 10 |
| 160 | each pair merges into one elongated vortex | 5 |
| 200 | five round vortices with filament halos | 5 |

The band-mode spectrum stays an exact comb of multiples of 5 (other modes at the 1e-5 noise level up to
t ≈ 60); dominant mode 5, then 10 (t ≈ 20–50), then 5. The five vortices hold about half of the initial band
area at T = 200 (the rest is filaments) and the pattern drifts along the band by one sector every ≈ 40 time
units (`analysis_mode5tips/e7_vortex_tracks.png`), so the vortices are not pinned to the lobes. Figures:
`preview/x5mode5tips_sequence.png`, `preview/x5mode5tips_spectrum.png`, `analysis_mode5tips/`.

`bulgetips` (stopped at t = 100, `preview/figures/x5bulgetips_sequence.png`): one lump per sector at t = 30,
then three to four vortices per sector (15–20 by t = 40) — not five either.

## Second round (user, 18:00: "the five fold perturbation did not work, it did not produce exactly 5 coherent
vortices. there's a way to perturb the equilibrium on the star while maintaining exactly 5 modes.")

At 0.05 d the seeds stay linear long enough for the thin band's fastest modes (m = 15–17, about three per
sector) to break each sector up. The classical way to force one vortex per wavelength (Kelvin–Helmholtz layers)
is a large amplitude: the fundamental turns nonlinear first, and the strain in the stretched band between the
forming vortices suppresses the short waves. Runs (T = 100): `mode5tipsA050`, `mode5tipsA100` (ℓ = 5 at the
lobe tips, edges displaced by 0.5 d and 1.0 d) and `bulgetipsA100`. Seed checks: area change ≤ 2e-6, no point
outside the wall, edge ratio 1.02–1.06 (RK4 forward/backward inverse error up to 1.3e-3 at 1.0 d; only the
backward map is used).

Result: `mode5tipsA050` / `mode5tipsA100` (sinuous) form five primary vortices first (t ≈ 10–15), then the
stretched band between them rolls up too (10–15 vortices by t ≈ 20–25): amplitude alone does not do it.

## Third round: varicose seeds give exactly five vortices

Varicose ℓ = 5 (`--varicose`, odd normal profile (η/w) g(η/w)): the two band edges move in opposite directions,
so each sector's band gathers into one short thick lump at the lobe tip with thin necks in between. At 0.45 d a
lump is about 1.9 d thick and 12 d long; its fastest wavelength (about 8 × 1.9 d ≈ 15 d) exceeds its length, so
it rolls up into exactly one vortex, and the necks carry too little charge to form coherent vortices.
`var5tipsA045` (`preview/figures/x5var5tipsA045_sequence.png`): five lumps at t = 10, five vortices from
t = 15, exactly five coherent vortices through t = 40 (no secondary roll-up), settled in the lobes by t = 30.
`var5tipsA045p6` (p = 6, T = 200, 72 min sharing the GPU; `preview/figures/x5var5tipsA045p6_check*.png`,
`_late.png`): five vortices from t = 15; the 0.1 d necks roll up into small satellites at t ≈ 40–55 (coherent
count 10–15 for a few frames), then five coherent vortices from t ≈ 57 to T = 200, one per lobe, with tiny
satellites by the wall. Active modes stay the 16 multiples of 5 until t ≈ 80, then the symmetry breaks slowly
(up to 60 active modes by t ≈ 155). In the article as `fig:e7-star-five`(b).
`var5tipsA025` (p = 4, 0.25 d): five structures until t = 30, then ten from t = 40 to T = 100 — with less
thickening the necks keep enough band to form five secondary vortices. Coherent counts (cores at 0.4 d):
0.45 d (p = 4) 5 at t = 0–40, 10 at t = 50, 5 at t = 60–100; 0.25 d 5 at t = 0–30, 10 at t = 40–100.
Study complete (`X5_DONE`, 20:28). If one of these is the case the user
remembers, rerun it at p = 6 (the E7 resolution) once the E7 campaign frees the GPU.
