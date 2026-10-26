# Consistent transfer of the E7 equilibria to HDG (2026-10-25)

## Question

The E7 controls start from the projected transfer ρ_h = Π_h W(φ_CG): the CG equilibrium's window density, L2-projected
onto the HDG space at the 24-point initial-projection rule. The evolution then derives its drift from φ_c(ρ_h), the
Oswald average of the p+1 postprocessed HDG potential of ρ_h (compatible drift). ρ_h is not steady under that
drift. Its steady transport residual contains W′(φ) ∇⊥(φ_c − φ_CG)·∇φ, the mismatch between the HDG and the source
CG potentials, which is largest at the re-entrant wall corners. The earlier onset analysis
(`../e7_preservation_20261024`) found that the seed of the natural instability is set at those corners.

## Method

- **Consistent transfer** (`hybridge/scripts/guiding_center/runtime/equilibrium_transfer.py`). It is the root of
  R(ρ) = ρ − Π_h W(φ_c(ρ)).
  - Plain fixed-point iteration diverges on the band's long edge modes (multipliers near 1/(kd) > 1). Newton is used
    instead: J = I − Π_h W′(φ_c) dφ_c/dρ, solved matrix-free by restarted GMRES in the L2 inner product.
  - Each product is one device HDG Poisson solve of a unit source, with the runs' fastest device options
    (raw-CUDA fb-hp-mg-pcg, legendre-modal traces, τ = 1, rtol 5e-11).
  - At the root, W(φ_c) is exactly steady under the drift ∇⊥φ_c, since the upwind-HDG residual of a continuous W(φ_c)
    vanishes. What remains is the transported projection error of W(φ_c).
- **Residuals** (`transfer_residuals`):
  - Window-law mismatch in L2, for the compatible potential and for the HDG potential.
  - The HDG semilinear residual −div q_h − W(φ_h) of the solved (φ_h, q_h, û_h), in the dual norm of the flux-jump
    Gram matrix. Its primal norm is Σ_K (‖∇u‖²_K + ‖q‖²_K + w_K ‖u − û‖²_∂K) with û = 0 on the wall and q
    independent of (u, û): w_K = 1 in the first checks (a norm on each mesh, not uniformly H¹₀ × L²), w_K = 1/h_K in
    `launch_gram_weights.py` (the discrete H¹₀ × L² norm, used in the article). The trace equations of the linear solve are
    left out, which makes the value independent of the trace basis. The same residual with source ρ_h verifies the
    solve.
  - The relative value is quoted against ‖ρ‖ in the same dual norm, and against √E_T. E_T, the HDG torsion energy,
    is the scale of the source's DOLFINx residual √(rᵀK⁻¹r/E_T).
  - The Gram trace Schur complement is assembled once and factored by PARDISO (`FactoredGramInverse`). It agrees with
    the matrix-free CG inverse to 2e-16 and needs 3 s per dual norm instead of more than 10 minutes.
- **Initial departure rate.** Under the semi-discrete evolution, dφ/dt(0) = −S F_h, with F_h the steady upwind-HDG
  residual and S the HDG Poisson solve. Its band modes are the initial rates Ȧ_m of the controls'
  `band_phi_mode_amplitudes`, computed from the initial state alone.

## Files

| Path | Content |
|---|---|
| `launch_checks.py` | Detached checks (record `checks_launch.json`; markers `<label>.done`, `CHECKS_DONE` / `CHECKS_FAILED.json`) |
| `<label>/<label>_*_transfer.json` | Residuals of both states, Newton records, change ρ* − ρ_h, band modes of the rates |
| `<label>/<label>_*_consistent_density.npz` | ρ* and ρ_h coefficients with the mesh |
| `analyze_checks.py` → `checks_summary.json` | Comparison table; predicted against measured Ȧ; predicted onset shift |
| `smoke/` | 20-step smoke of `run_band_evolution --equilibrium-transfer consistent` (resolved Pac-Man) |
| `attempts/` | First launches: CG Gram inverse too slow, then a variable-name clash; both were stopped |
| `logs/` | Driver logs per check |

## Results

### Resolved Ugly Pac-Man (158,358 triangles, p = 6; control `res_uglypacman_p6`, onset t = 48)

| | projected | consistent |
|---|---|---|
| window-law mismatch, φ_c (L2) | 1.8e-4 | 1.1e-12 (two Newton steps, 89 GMRES products, 22 s) |
| HDG semilinear residual, Gram dual / ‖ρ‖ | 1.4e-4 | 4.6e-10 |
| same over √E_T (DOLFINx scale; source 4.1e-15) | 4.6e-4 | 1.5e-9 |
| linear Poisson residual, Gram dual / ‖ρ‖ (solve check) | 1.6e-11 | 1.6e-11 |
| steady transport residual, L2 / ‖ρ‖ | 5.9e-4 | 5.9e-4 |
| steady transport residual, Gram dual / ‖ρ‖ | 3.4e-6 | 6.3e-7 |
| potential rate, energy norm (relative) | 3.5e-6 | 7.3e-7 |
| Ȧ₁₆ (dominant mode) | 2.16e-7 (measured 2.17e-7) | 3.2e-12 |

- **Predictor validated.** Predicted Ȧ_m over measured Ȧ_m for modes 1–40 has median 0.998 (range 0.89–1.09).
- **Seed removed.** The consistent transfer removes the seed of the dominant mode by a factor of about 7e4. The
  predicted onset shift is +81 time units. The L2 steady residual is unchanged and is grid-scale: it does not
  project onto the band modes.

### Corner resolution (corner-refined Pac-Man source, ε/δ = 0.08)

| | refined corners, projected | refined corners, consistent | uniform corners (k0), projected | k0, consistent |
|---|---|---|---|---|
| triangles | 215,246 | | 209,996 | |
| window-law mismatch, φ_c (L2) | 3.5e-7 | 1.6e-12 (1 Newton step) | 5.0e-5 | 1.3e-12 (2 steps) |
| HDG semilinear residual, Gram dual / ‖ρ‖ | 2.9e-7 | 2.0e-11 | 4.1e-5 | 1.6e-11 |
| steady transport residual, Gram dual / ‖ρ‖ | 9.5e-9 | 6.2e-9 | 1.0e-6 | 6.3e-9 |
| potential rate, energy norm | 7.7e-9 | 4.1e-9 | 9.4e-7 | 4.1e-9 |
| Ȧ of the dominant mode | 3.4e-10 (m = 18; measured 3.39e-10) | 1.0e-14 | 5.4e-8 (m = 17; measured 5.45e-8) | 2.0e-14 |

- **The projected transfer's error is corner error.** Refining the corners lowers the window-law mismatch from 5e-5
  to 3.5e-7, and the seed rate by about 160.
- **With the consistent transfer the corners no longer matter.** The corner-free mesh gives the same remaining
  residuals and rates as the corner-refined one, and the dominant mode starts at round-off on both.
- **The predictor holds on all three Pac-Man meshes.** Predicted over measured Ȧ_m has median 0.998 for modes 1–40.

### Sharpest band, mesh b (279,662 triangles; outer edge narrower than one cell; control `mb_ctrl_b_p6_dt0p025`, onset t = 42)

| | projected | consistent |
|---|---|---|
| window-law mismatch, φ_c (L2) | 2.0e-4 | 3.2e-12 (two Newton steps, 102 products) |
| HDG semilinear residual, Gram dual / ‖ρ‖ | 1.3e-4 | 7.8e-8 |
| steady transport residual, L2 / ‖ρ‖ | 0.18 | 0.18 |
| steady transport residual, Gram dual / ‖ρ‖ | 1.05e-4 | 1.05e-4 |
| Ȧ₂₅ (dominant mode) | 1.375e-7 (measured 1.377e-7) | 1.8e-10 |

- **The control settles what the check could not.** The check's consistent Ȧ₂₅ = 1.8e-10 is the predictor's floor:
  the rate solve's Poisson tolerance (5e-11) applied to a steady residual that is large and grid-scale (0.18 in L2).
  The consistent control measures A₂₅(1) = 2.0e-13, against 1.4e-7 projected, and A₂₀(4) = 9.5e-12, against 1.5e-6.
  The growing modes start near round-off here too.
- **What the unresolved edge sets.** The grid-scale departure is unchanged by the transfer: e_ρ(1) = 1.2e-3 and
  e_φ(1) = 6.8e-7, against 7.3e-7 projected, with an undershoot of −6.7e-3. It does not reach the band modes.
- **Predictor accuracy.** For the five-fold modes carrying the projected transfer's corner seed, predicted over
  measured is 0.98–1.12. The other modes sit four orders lower, at about 1e-11, below what the predictor resolves
  with that tolerance (0.15–40). The predictor is accurate wherever the forcing exceeds its floor, which is about
  1e-11 of the steady residual's scale.

### Summary

The consistent transfer makes the initial state a discrete equilibrium of the HDG evolution up to the transported
projection error of the window:
- **Resolved edges (ε/δ = 0.08).** The growing modes start from round-off on every Pac-Man mesh, with or without
  corner refinement. The consistent control's onset moves from t = 48 to about t = 140 (`runs/`).
- **Sharp edge.** The growing modes start near round-off as well. The edge's projection error only sets the
  grid-scale plateau of e_ρ and e_φ.
- **Corners.** They no longer matter.
- **Precision claim.** Preservation of an unstable band lasts t* ≈ γ⁻¹ ln(A*γ/Ȧ), where Ȧ, the initial band-mode rate
  of the dominant mode, is computed before the run from the band modes of −S F_h.

## Article (2026-10-25)

Section 7.2 is rewritten around the transfer. Its paragraphs:
- What preservation means.
- The transfer sets the seed.
- A consistent transfer.
- Preservation from the consistent transfer.
- The CG evolution of the same source, with a `\pending` for the consistent CG start.
- The onset law and what it allows.
- The reach of the geometry.

Its floats are `tab:e7-transfer` (`tables/e7_transfer.tex`), `fig:e7-transfer` (`make_figures_transfer.py`),
`tab:e7-midband-setup`, `fig:e7-onset-prediction` (`../e7_preservation_20261024/make_figures_72.py prediction`) and
`fig:e7-developed`.

The projected-transfer controls, with their figures and tables, moved to supplement S8. The introduction bullet, the
discussion and the section's source comment are updated, and Section 7 spans pages 25–29. The previous 7.2 is in
`article_drafts/section_7_2_projected_before_rewrite.tex`.

To finish once the runs end:
1. Rerun `analyze_checks.py`, `analyze_controls.py` and `../e7_preservation_20261024/analyze_onset.py`.
2. Regenerate the two figures and the table. The Pac-Man and k0 consistent stars appear in `fig:e7-onset-prediction`
   once their runs write a summary.
3. Fill the `\pending` with the consistent CG runs.

### Restructured (2026-10-25, user: the transfer is too technical for 7.2)

Section 7.2 now has:
- a remark on the initial state (the discrete equilibrium of the scheme; the projection preserves about three times
  shorter);
- three conclusions: the onset time, re-entrant corners (their numerical role as the seed of an inconsistent start,
  their physical role pending the corner-free runs), and the modes the bands select;
- floats `tab:e7-natural` and `fig:e7-natural` (`make_figures_natural.py`), `fig:e7-onset-prediction`,
  `tab:e7-midband-setup` and `fig:e7-developed`.

The transfer moved to supplement S8: the residual table, the predictor figure and the CG/SUPG comparison.
`launch_natural.py` queues the consistent controls for the band at 3.65d, the plain star, the horseshoe and ITER.
They will fill the two `\pending` placeholders of 7.2 and the table. The previous version is in
`article_drafts/section_7_2_transfer_version.tex`.

### Placement study and the h_K⁻¹ Gram (2026-10-25, later)

- **Gram weight.** `launch_gram_weights.py` (detached, done: `GRAM_ELEMENT_DONE`) reran the four checks with
  `--gram-jump-weight element --consistent-npz` (no Newton; the stored and the new projections agree exactly) into
  `<label>_element/`. `analyze_checks.py` adds the values with the suffix `_element`; `make_figures_transfer.py`
  puts them in `tab:e7-transfer`. Relative HDG residuals, projected → consistent: Pac-Man 1.4e-4 → 5.2e-9 (unit
  weight 4.6e-10: its consistent residual is at the grid scale), sharpest band 1.4e-4 → 8.6e-8, refined corners
  2.8e-7 → 2.2e-11, refined source on the uniform mesh 4.1e-5 → 1.9e-11. No conclusion changes.
- **7.2 rebuilt on the placements.** `make_figures_natural.py` now writes `tab:e7-natural` (four domains and the seven
  undercut-star bands of `../e7_placement_20261025`, reading `runs_t300/` where a run has finished), `fig:e7-natural`
  ((a) dominant-mode amplitude, (b) rate against the clearance with the planar strip next to a wall, (c) measured
  against predicted onset for all controls, (d) spectra of three undercut-star bands) and, with `--developed`,
  `fig:e7-developed` (five undercut-star bands at the last saved state before the onset t(1e-3) and at the
  roll-up; since 2026-10-26, rasters at 600 dpi). `fig:e7-onset-prediction` left the article
  (its panel (a) is now panel (c); the projection spectra are in S8, `fig:e7-onset-law`d); `tab:e7-midband-setup`
  moved to S8 with the sharpest band. The previous Section 7 is in
  `article_drafts/section_7_before_placement_rewrite.tex`.
- **S8.** Norm defined with h_K⁻¹; the CG paragraph rewritten with `tab:e7-cg` (`tables/e7_cg_baseline.tex`,
  `../e7_cg_baseline_20261025/analyze_baseline.py --table`) and both CG starts in `fig:e7-transfer`b.

