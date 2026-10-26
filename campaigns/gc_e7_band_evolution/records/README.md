# E7: preservation and band-mode instability of the Figure 14 equilibria

Plan: `~/.claude/plans/optimized-orbiting-lecun.md` (approved 2026-10-19); article PLAN.md, E7.

## Question

1. How well does the HDG scheme preserve the four general-geometry equilibria of Figure 14
   (star, Ugly Pac-Man, ITER wall, horseshoe)? Each **control** starts from the L² projection of
   W(φ*) of the CG6 equilibrium on its own mesh.
2. Which diocotron-like instabilities develop on these bands, and how does the wall shape them?
   Each **seed** run starts from a band-mode displacement of the same equilibrium.

## Seed (geometry-aware, area-preserving)

- Coordinates come from the band's own middle equipotential (`contours.npz` of the stage).
  - θ is the flow angle: 2π times the time of flight along the contour at the drift speed |∇φ*|, divided by the circulation period.
  - η is the signed normal distance, positive outward.
- Stream function χ = A g(η/w) sin(ℓθ + ψ) c_wall(x).
  - g(u) = (1 − u²)³ on |u| < 1, with half-width w = 2d (d is the mean band thickness).
  - c_wall is a C² cutoff that freezes points within d/2 of the wall.
  - χ is tabulated at spacing d/20 and represented by an interpolating cubic B-spline, so ∇⊥χ is exactly divergence-free.
- Initial density ρ₀ = W(φ*(Φ₋₁(x))), where Φ is the time-1 flow of ∇⊥χ (RK4, 4 steps).
  - This is a finite area-preserving displacement: charge and every Casimir are kept, and ρ₀ ∈ [0, 1].
- Mode ℓ is the vortex-sheet most-unstable estimate k d ≈ 0.8, i.e. ℓ ≈ 0.8 L/(2π d):

  | Domain | ℓ |
  |---|---|
  | star | 16 |
  | Ugly Pac-Man | 16 |
  | ITER | 18 |
  | horseshoe | 20 |

- The phase ψ puts the largest outward displacement at the band point nearest the wall.
- The amplitude A makes the largest normal edge displacement 5 % of d.
- Seed checks (host, before any run) are in `seed_check/`.

## Numerical configuration (control and seed identical)

- **Mesh:** the embedded band-graded CG6 mesh (215k–264k triangles).
- **Discretization:** p = 6 for density and Poisson, with the compatible drift (Oswald average of the p+1 postprocessed potential).
- **Poisson and transport:** τ = 1 with no retries, conflict-averaged upwind.
- **Time stepping:** SI-BDF2, Δt = 0.05, T = 200 (4000 steps).
- **Quadrature:** volume rule (3p+2)//2 = 10 per direction; initial projections 24 per direction.
- **Device:** fastest recorded device configuration, Poisson absolute tolerance 5e-14 (4.5e-11 of the right-hand side; 1e-14 sits under the round-off floor of the first solve on these bands, see `runs_failed.jsonl`).
- **Saved states:** ρ_h and φ_h every 2.5 up to t = 60, then every 5.

## Diagnostics

- **Every step (JSONL):**
  - mass, energy, enstrophy and L² budget;
  - min/max and positivity;
  - drift from the projected equilibrium (`diocotron_rho_eq_relative_l2`, `diocotron_phi_eq_relative_l2`);
  - band-mode spectra A_m, m = 0…80, of ρ_h and φ_h along θ (`band_*`), with the number of active modes (≥ 1 % of the largest), the dominant modes and the participation number.
- **Control only:** transfer metrics `equilibrium_*`.
- **Offline (`analysis/`):** growth-rate fits, control-versus-seed spectra, vortex census on the saved states, snapshot figures and movies.

## Running

```bash
cd ~/src/gc-hdg-article/run_outputs/e7_band_evolution_20261019
setsid nohup taskset -c 12-23 ./launcher.py > driver_stdout.log 2>&1 < /dev/null &
```

- `driver.json` records the process group and the stop command (`kill -- -<pgid>`).
- Restarting resumes and skips finished runs.
- `E7_DONE` marks completion; `E7_FAILED` marks a stop.
- **GPU lock:** this is an accuracy campaign, so it never takes the lock. Before each run it waits until no performance campaign holds the lock and 24 GB of device memory are free.
- **Order of runs:** a 20-step smoke pair on the star (`runs/smoke`), then star, Pac-Man, ITER and horseshoe, each control before its seed.

## Results (campaign complete 2026-10-19 23:20, `E7_DONE`)

All eight production runs ok (`runs.jsonl`; failed attempts in `runs_failed.jsonl`, logs in `runs/logs/failed/`).
Analysis: `analysis/` (`e7_summary.md|json`, spectra, modes, snapshots, vortex tracks), from
`.venv/bin/python -m scripts.guiding_center.diagnostics.analyze_band_evolution --runs runs --output analysis`.
Article floats (§7.2): `scripts/guiding_center/diagnostics/plot_band_evolution_article.py`, commands in
`~/src/gc-hdg-article/figures/README.md`.

| Domain | ρ drift t = 1 | φ drift t = 50 | max A_m t = 1 → 50 | max A_m reaches 1e-3 | seeded γ (window) / strip | control fastest m (γ) / strip | vortices max → T | enstrophy at T (control / seed) |
|---|---|---|---|---|---|---|---|---|
| Star | 1.2e-2 | 1.1e-5 | 3.0e-8 → 2.6e-5 | t = 69 | 0.201 (2.0–7.5) / 0.201 | 17 (0.186) / 0.200 | 16 → 8 | −30 % / −40 % |
| Ugly Pac-Man | 1.1e-2 | 2.1e-5 | 2.1e-8 → 9.9e-5 | t = 63 | 0.201 (2.0–7.1) / 0.201 | 16 (0.184) / 0.201 | 16 → 8 | −31 % / −48 % |
| ITER | 1.2e-2 | 6.0e-6 | 3.3e-8 → 6.8e-5 | t = 64 | 0.209 (2.0–7.6) / 0.201 | 19 (0.195) / 0.200 | 18 → 11 | −27 % / −41 % |
| Horseshoe | 1.1e-2 | 1.9e-5 | 2.5e-8 → 8.6e-5 | t = 62 | 0.212 (2.0–6.9) / 0.201 | 19 (0.213) / 0.201 | 20 → 8 | −37 % / −54 % |

All runs: mass drift ≤ 4e-13, L² identity defect ≤ 1e-15, energy drift 1.1–2.2e-3, min ρ_h −0.53 … −1.10.
The ρ drift at t = 1 is the relaxation of the projected sharp edges (angle independent); the angular departure
max_{m≥1} A_m is what the instability grows from. Geometry: vortices collect at the hairpin turns of the
horseshoe band and pause at the strongest convex bend of the Ugly Pac-Man band (`analysis/e7_vortex_tracks.png`);
vortices past the Pac-Man mouth corners are strongly strained and the transport solve needs 120–180 iterations per
step there (25 elsewhere); ITER filaments reach the divertor legs. The perturbed runs fill the band spectrum first
on the Ugly Pac-Man and the horseshoe and last on the five-fold star. Five-fold seeds: `../e7_five_sites_20261019/`.

Incidents: Poisson atol 1e-14 → 5e-14 (smoke), seed wall points frozen (`uglypacman_seed`), Poisson rtol 1e-11 →
5e-11 (`iter_control`); see PLAN.md E7. Step cost with the GPU to itself: 0.39–0.42 s (horseshoe/ITER/star
control); runs sharing the GPU with the five-fold study took 45–96 min.
