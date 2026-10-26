# CG baseline of equilibrium preservation (2026-10-25)

## Question

Is a window equilibrium preserved when it is evolved in the discretization it was computed in, without the HDG
transfer? The run uses DOLFINx CG Poisson with CG/SUPG transport and SI-Euler: backward-Euler SUPG transport with
the previous step's potential (`hybridge-diocotron/projects/diocotron/dolfinx/guiding_center/supg.py`).

## Setup

- **Source.** The resolved Ugly Pac-Man source, ε/δ = 0.08 (`equilibrium_dt080_20261016/cg{4,6}/uglypacman`). It is
  evolved on its own 158,358-triangle mesh in its own Lagrange space, with CG4 (1,275,089 dofs) and CG6 (2,862,781
  dofs).
- **Initial state.** φ_0 = φ_CG, and ρ_0 = I_h W(φ_CG), the nodal interpolant of the source's window. The converter
  is `adapt_checkpoint.py`.
- **Hand-off.** The Poisson solution of ρ_0 differs from φ_CG by 2.0e-7 in L2 (9.3e-7 in H1) at CG4. This is the
  CG analogue of the HDG–CG potential gap: 9.9e-6 for the projected HDG transfer at p = 6, and 6.3e-6 for the
  consistent one.
- **Solver preset.** The recorded optimized preset: CG/BoomerAMG Poisson, BiCGStab with block-Jacobi ILU(0)
  transport, SUPG scale 0.1, no CIP, quadrature degree 16, 20 MPI ranks.
- **Time step.** dt = 0.05 and T = 100, as in the HDG controls `res_uglypacman_p{4,5,6}_dt0p05`. Backward-Euler damping
  of the band waves is negligible here: T_c = 177, so ω²dt/2 ≈ 0.008, against the growth rate 0.14.
- **Diagnostics.** `runs/<label>/diagnostics.csv`, every 0.1 time units: e_ρ, e_φ in L2 and in the energy norm,
  measured from the Poisson solution of ρ_0. Also the advective defect, mass, energy and the ρ range.
- **Snapshots.** ρ and φ as portable v2 checkpoints at t = 0, 1, 2, 5, 10, …, 100. The t = 0 file also holds
  φ_initial.

Code changes in the diocotron worktree: `supg.py` gained a nearest-neighbour fallback for the checkpoint nodal match
(the quantized keys fail on millions of nodes), e_φ in L2 and energy norm, and `--snapshot-times`.

## Files

| Path | Content |
|---|---|
| `adapt_checkpoint.py` | catalog φ-only checkpoint → `hdgfem_equilibrium_v2` (ρ = I_h W(φ)) |
| `uglypacman_cg{4,6}_supg.npz` | converted inputs |
| `launch_baseline.py` | detached CG4 → CG6 chain (`baseline_launch.json`, markers `CG4_DONE`, `CG6_DONE`, `BASELINE_DONE`) |
| `analyze_baseline.py` → `baseline_summary.json` | band modes of φ(t) − φ_0, the same definition and band tables as the HDG controls; seed, onset, rate |
| `smoke/` | 8-step timing check: CG4 at 1.2 s per step on 20 ranks |

## Consistent start (`consistent_start.py`, `launch_consistent.py`)

The DOLFINx equilibrium solves ∫∇φ·∇v = ∫W(φ)v with the window at the quadrature points. The simulation instead
evolves ρ in the same Lagrange space, starting from the nodal interpolant ρ₀ = I_h W(φ_CG), whose Poisson solution is
not φ_CG. With nodal interpolation, ρ = I_h W(φ) means ρᵢ = W(φᵢ) at every node. The simulation's own steady state
therefore solves

    K φ = M₀ W(φ),

with K the stiffness matrix with zero Dirichlet rows and M₀ the mass matrix without its boundary rows. Newton from
φ_CG with J = K − M₀ diag(W′(φ)), factored by MUMPS:
- **CG4.** The residual goes 7.7e-5 → 6.1e-11 → 7.9e-12 (round-off floor). The Poisson solve of ρ₀ returns φ* to
  2.4e-13, against 2.0e-7 for the native start. φ* differs from φ_CG by 2.0e-7.
- **CG6.** The residual goes 5.1e-6 → 2.8e-11 (round-off floor). The Poisson solve of ρ₀ returns φ* to 3.0e-13,
  against 6.1e-9 for the native start. φ* differs from φ_CG by 6.2e-9.
- **Runs.** `launch_consistent.py` runs CG4 and CG6 from these states after the native chain, each identical to its
  native twin except for the start.

## Poisson preconditioner (bench/, CG6 Pac-Man, 2.86M dofs, 20 ranks, 6 steps each)

| Poisson solve | hierarchy setup | cold solve (rtol 1e-10) | warm solve per step | whole step |
|---|---|---|---|---|
| CG + BoomerAMG (preset) | below 1 s | 15 its, 7.5 s | 1 it, 1.17 s | 3.67 s |
| CG + GAMG, no graph filtering | | 104 its, 20.6 s | 1.15 s | 3.50 s |
| CG + GAMG, threshold 0.02 | | 82 its, 17.2 s | 0.84 s | 3.34 s |
| **CG + GAMG, threshold 0.05** | 2.8 s | 54 its, 12.2 s | 2 its, 0.69 s | 3.20 s |
| CG + GAMG, threshold 0.05, 3 smoothing steps | 2.8 s | 44 its, 13.4 s | 0.96 s | 3.49 s |
| CG + GAMG, threshold 0.1 | 19.5 s | 46 its, 31.2 s | 1.00 s | 3.50 s |
| CG + GAMG, threshold 0.15 | 151 s | 129 its, 207 s | 9 its, 4.66 s | 7.19 s |

- **Where the Poisson time goes.** Form creation (2 ms) and assembly (21 ms) are negligible; the warm solve is the
  preconditioner application.
- **The hierarchy is built once.** PETSc's event log shows 7 PCSetUp calls in every variant: the Poisson hierarchy
  once, and the transport ILU at each of the 6 steps.
- **CG6 uses CG + GAMG with threshold 0.05.** It was relaunched with this setting (the BoomerAMG run, stopped at
  t = 23, is in `attempts/`). Transport, at 2.45 s per step (0.54 s assembly, 1.87 s ILU(0) + BiCGStab), is now the
  larger part of the step.

## Results (CG4; CG6 running)

| t | CG4 e_φ | HDG p = 4 e_φ (projected transfer) | ratio |
|---|---|---|---|
| 1 | 4.8e-8 | 9.7e-7 | 20 |
| 10 | 3.9e-7 | 1.4e-5 | 37 |
| 25 | 2.0e-6 | 1.3e-4 | 64 |
| 50 | 7.4e-5 | 2.3e-3 | 31 |

- **Same physics, smaller seed.** The growth rate matches the HDG controls (e_φ doubles every 5 time units, about
  0.14), and the CG run reaches the HDG departure level about 22–25 time units later.
- **The seed is the transfer.** Without the HDG projection, the seed is 20–60 times smaller. This matches the
  potential hand-off ratio (9.9e-6 / 2.0e-7 ≈ 50).
- **The consistent transfer goes much further.** It starts the HDG p = 6 control with the growing band mode at
  round-off (`../e7_transfer_20261025`).

### Band modes (`analyze_baseline.py`, the HDG controls' definition)

CG4 started before snapshots carried φ_initial, so its modes are measured from the saved potential. The low modes
(A₂ ≈ 2.7e-7, constant) hold the static Poisson hand-off. The growing modes m = 13–22 are only marginally affected
(at most A_m(1) ≈ 1–4e-8).

| run, same source and mesh | seed of the growing mode at t = 5 | dominant mode, rate | t(1e-4) | t(1e-3) |
|---|---|---|---|---|
| HDG p = 6, projected transfer | A₁₆ = 1.3e-6 | 16, 0.137 | 29.3 | 48.0 |
| CG4/SUPG, no projection | A₂₀ = 5.4e-8 (6.2e-8 back-extrapolated from the growth) | 20, 0.164 | 50.3 | 63.5 |
| HDG p = 6, consistent transfer | A₁₈₋₂₀ ≈ 1e-13 | 18–20, 0.17–0.18 | about 117 | about 140 |

CG/SUPG without projection seeds about 25 times below the projected HDG transfer. Its own transfer is not exact
either: ρ_0 = I_h W(φ_CG), and the Poisson hand-off is 2e-7. The consistent HDG transfer seeds five to six orders of
magnitude below CG/SUPG.

## All four runs (2026-10-25, `analyze_baseline.py`, `--table` → `tables/e7_cg_baseline.tex`, supplement `tab:e7-cg`)

| run | hand-off ‖∇(φ_h(0) − φ_start)‖ | two steps ‖∇(φ_h(0.1) − φ_h(0))‖ | ‖u·∇ρ₀‖/‖ρ₀‖ | seed max A(5) | mode | rate | t(1e-3) |
|---|---|---|---|---|---|---|---|
| CG4, interpolant | 9.3e-7 | 7.0e-7 | 7.8e-3 | 2.6e-7 | 20 | 0.166 | 64 |
| CG4, discrete steady state | 1.1e-10 | 7.0e-7 | 7.8e-3 | 2.0e-7 | 20 | 0.162 | 61 |
| CG6, interpolant | 2.8e-8 | 2.6e-8 | 6.4e-4 | 5.0e-9 | 19 | 0.162 | 87 |
| CG6, discrete steady state | 5.7e-9 | 2.6e-8 | 6.4e-4 | 7.0e-9 | 20 | 0.171 | 84 |

(Hand-off and two-step change relative to ‖∇φ_h(0)‖. The CG6 steady-state hand-off is limited by the run's GAMG
tolerance; Newton with MUMPS reached round-off.)

- **The consistent start does not lower the seed.** Removing the Poisson hand-off (×10⁴ at CG4) changes the seed by
  less than 1.5 and the onset by three time units. The first two steps move φ by the same amount from both starts.
- **Correction of the earlier reading.** The native seed is not the representation mismatch: the hand-off and the
  transport response are both interpolation errors of the same window, of the same size.
- **What seeds the CG run** is the transport step's response to the interpolated window, whose advective defect
  ‖u·∇ρ₀‖ does not vanish (the undershoot after two steps is −2.7e-4 at CG4, −2.2e-5 at CG6). It falls with the
  degree (seed ÷30–50 from CG4 to CG6).
- **Hypothesis** for the difference with HDG: the HDG start is an L² projection, whose error is orthogonal to the
  polynomial space and barely reaches the smooth band modes; the CG start is a nodal interpolant. Tested and confirmed
  on 2026-10-26 by the t = 0 predictor below.

## t = 0 predictor: interpolant against L² projection (2026-10-26, `t0_predictor.py`, `t0_modes.py`)

No time stepping. For each start, `t0_predictor.py` (DOLFINx, 16 ranks; 94 s at CG4, 181 s at CG6) computes
φ₀ = K⁻¹Mρ₀ and the potential rate φ' = K⁻¹Mρ', with ρ' from the transport matrices before any step:
- **Galerkin.** Mρ' = −Cρ₀.
- **SUPG.** (M + M_τ)ρ' = −(C + C_τ)ρ₀, with the runs' transient τ at dt = 0.05.
- **First steps.** The first backward-Euler increment of both schemes, with the potential the run starts from (one
  MUMPS solve; GMRES with block-Jacobi ILU(0) stalls on the step matrix without SUPG).
- **Band modes.** `t0_modes.py` takes them with the runs' definition, normalized by ∫|φ₀|g of each start.
  `t0/cg{4,6}.npz` holds the fields, `t0/cg{4,6}.json` the norms and solver records, `t0_predictor.json` the modes.

The three starts:
- **I_hW(φ_CG):** the runs cg4, cg6.
- **I_hW(φ\*):** the runs cg4_consistent, cg6_consistent.
- **P_hW(φ_CG):** the L² projection onto the whole Lagrange space, with the source's window and quadrature (degree
  32). The source solves ∫∇φ·∇v = ∫W(φ)v, so the Poisson solution of P_hW(φ_CG) is φ_CG itself: the hand-off is
  3.5e-12 at CG4 and 1.9e-11 at CG6. This start was not run.

Initial rate of the final dominant mode (m\* = 20 at CG4, 19 at CG6), relative to ∫|φ₀|g:

| | start | Galerkin | SUPG | first step, SUPG | measured A_m\*(1) | ‖u·∇ρ₀‖/‖ρ₀‖ |
|---|---|---|---|---|---|---|
| CG4 | I_hW(φ_CG) | 2.58e-8 | 2.58e-8 | 3.83e-8 | 2.66e-8 | 7.8e-3 |
| CG4 | I_hW(φ\*) | 3.83e-8 | 3.83e-8 | 3.83e-8 | 3.82e-8 | 7.8e-3 |
| CG4 | P_hW(φ_CG) | 1.68e-12 | 1.87e-12 | 4.62e-12 | not run | 6.6e-3 |
| CG6 | I_hW(φ_CG) | 4.76e-10 | 4.76e-10 | 6.98e-10 | 4.54e-10 | 6.4e-4 |
| CG6 | I_hW(φ\*) | 6.99e-10 | 6.99e-10 | 6.98e-10 | 6.49e-10 | 6.4e-4 |
| CG6 | P_hW(φ_CG) | 4.0e-15 | 4.6e-15 | 9.3e-15 | not run | 4.8e-4 |

- **The predictor works for CG as for HDG.** The semi-discrete rate matches the measured A_m\*(1) to within 7 % in
  all four runs. Over the growing modes (at least a tenth of the largest amplitude at t(1e-4); m = 9–25) the median
  ratio is 0.94–1.04. The interpolant runs' first step uses φ_CG, whose drift differs from that of φ₀; every later
  step uses Poisson(ρ), so the semi-discrete rate with φ₀ is the one that matches.
- **The start sets the seed.** The projection has nearly the advective defect of the interpolant. Its rate is about
  2×10⁴ times lower at CG4, and 10⁵ times lower at CG6, where it is at round-off: the HDG discrete equilibrium
  measures Ȧ ≈ 1e-14 (`tab:e7-transfer`).
- **SUPG changes little.** It raises the projection's rate by 10–15 % (semi-discrete) and by a factor of 2 in the
  first backward-Euler step, which is still 4–5 orders below the interpolants.
- **Why.** For continuous v, div u = 0, u·n = 0 on the wall and u·∇W(φ) = 0 give
  ∫(u·∇ρ₀)v = −∫(ρ₀ − W(φ)) u·∇v. The projection error is orthogonal to the Lagrange space and reaches the smooth band
  modes only through the part of u·∇v outside it (the jumps of ∇φ_h). The interpolation error is not orthogonal to
  the space and reaches them directly. The HDG discrete equilibrium is an L² projection of the same kind.
- **Correction.** Calling I_hW(φ\*) the "discrete steady state" was wrong: it removes the Poisson hand-off but not
  the interpolation error of the transport. With the source's quadrature, the Poisson-consistent start of the CG
  evolution is P_hW(φ_CG), which needs no Newton solve.
- **Onset formula.** t(1e-3) = ln(1e-3 γ/Ȧ)/γ gives 52.9, 50.5, 78.5 and 76.1 for the four runs, which measured
  63.5, 60.8, 86.8 and 84.0. The formula is early by 8–11 time units because the back-extrapolated seeds are
  (0.9–1.7) Ȧ·1, not Ȧ/γ: the band waves oscillate at ω ≈ 0.6 > γ. With the same offset, the projection would reach
  1e-3 at about t ≈ 120 at CG4. At CG6 it would start at round-off, like the HDG discrete equilibrium (onset 140).
- **Article.** Supplement S8: new paragraph "The start sets the CG seed", `tab:e7-cg` regenerated by
  `analyze_baseline.py --table` (predicted rate, measured A(1), the two projection rows), and the `fig:e7-transfer`
  legend.
- **Not run (offered).** CG/SUPG from P_hW(φ_CG). A 20-step CG4 run (about 1 min) would check the predicted
  A_20(1) ≈ 1.9e-12. Full runs to T = 150 would take about 1 h at CG4 and 2.7 h at CG6 (20 ranks).
