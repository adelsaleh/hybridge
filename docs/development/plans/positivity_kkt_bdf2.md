# Positivity-Preserving SI-BDF2 by a KKT Projection (Guiding-Center Positive Turbulence)

Status: Phases 1 and 2 implemented 2026-10-05; the manufactured order
gates passed ([results](../../research/solver_studies/positivity_order_gates_2026_10_05.md)).
The full-size positive-case check and Phase 3 are open. Evidence: the
[static study](../../research/solver_studies/positivity_projection_static_2026_10_05.md).
Roadmap entry: the Guiding-Center section of [`TODO.md`](../../../TODO.md).

## Goal

Keep the guiding-center density nonnegative in the positive-turbulence case
with SI-BDF2 without lowering the order of convergence (second order in time,
p + 1 in space) and at a small fraction of the step cost.

The efficiency payoff is the time step. The published positive run needed
dt = 0.000390625, a quarter of the signed run's step and an eighth of the
preset's 0.005, mainly to keep undershoots small. At t = 0.5 the minimum
density is −0.814, −0.25 and −0.00035 for dt = 0.0015625, 0.00078125 and
0.000390625. If positivity holds by construction, dt can be chosen on accuracy
alone.

## Diagnosis

One SI-BDF2 step solves the linear transport problem

```text
(M + A(β*)) ρ* = M S,    S = (4ρⁿ − ρⁿ⁻¹)/3,    β* = (2dt/3)(2vⁿ − vⁿ⁻¹),
```

then the endpoint Poisson problem with the new density. History is committed
only after both solves succeed (`scripts/guiding_center/time_schemes/si_bdf2.py`,
`si_euler.py`).

The undershoots have three sources:

- **Temporal, dominant at the published resolution.** S < 0 wherever
  ρⁿ⁻¹ > 4ρⁿ, just behind a filament moving into near-zero density. No
  monotone implicit operator can make ρ* nonnegative from a negative source.
  Deep undershoots (< −1e-3) at t = 0.5 fall from 5,228 elements to 833 to 4
  as dt halves twice.
- **Spatial.** High-order DG ringing at filament and blob tails: about 7,000
  elements between −1e-6 and −1e-3, independent of dt.
- **Long-time accumulation.** In under-resolved turbulence, 94,000 elements
  are below −1e-3 at t = 6.4.

About 40% of the violating elements have a negative cell mean. Cell-average
scaling limiters therefore cannot restore positivity alone; mass has to move
between elements.

## Formulation

Positivity is imposed at a point set S of each element, not everywhere. The
default S is the volume quadrature points plus the p-lattice (vertices, edge
and interior nodes): 61 points at p = 6.

| Choice | Measured spatial rate at p = 6 |
|---|---|
| Plain L2 projection | 6.9 |
| Point constraints at S | 7.0 (error within 1–6% of the plain projection) |
| Bernstein coefficients ≥ 0 (positivity everywhere) | ≈ 2.5 |

Bernstein constraints are therefore rejected. S is a configuration choice.
Adding the (2p+2)-lattice that the diagnostics sample makes those points exact
too, at about three times the element work, and keeps the order.

### Coupled KKT

The van der Vegt–Xia–Xu style coupled form adds Lagrange multipliers to the
BDF2 system itself:

```text
(M + A) ρ − M S = Bᵀλ + ξ M1,   λ ≥ 0,   Bρ ≥ 0 at S,   λ ∘ Bρ = 0,   1ᵀM(ρ − ρ*) = 0.
```

It is solved by semismooth Newton on min(λ, Bρ) = 0. In HDG the constraints
are element-local, so they can live inside static condensation, which leaves
the trace system semismooth in the trace. Each Newton iteration then means:

- updating the condensed operator on elements with active constraints;
- an AMGX setup, because the operator changes;
- one global transport solve.

Two to five iterations would add one to four transport solves per step. Mass
would move through numerical fluxes and keep local conservation, but the cost
and the intrusion into the raw-CUDA assembly are high.

### Decoupled KKT projection (recommended)

The operator-splitting form of the same Lagrange-multiplier idea (Cheng–Shen)
keeps the existing unconstrained solve and projects its result:

```text
ρⁿ⁺¹ = P(ρ*) = argmin ½‖ρ − ρ*‖²_M   s.t.  ρ ≥ 0 at S,   ∫ρ = ∫ρ*.
```

With the orthogonal `dub_orth` basis, M is diagonal per element, so P splits
into element problems tied together only by the mass constraint. The
implemented variant keeps local conservation wherever it can:

1. **Nonnegative element mean.** Take the closest polynomial that is ≥ 0 at S
   and keeps the element mean. This is always feasible (the constant mean),
   exactly conservative, and the minimal-norm counterpart of Zhang–Shu
   scaling.
2. **Negative element mean.** Take the closest polynomial that is ≥ 0 at S,
   with no mean constraint. This adds a small mass δ_K > 0.
3. **Mass return.** Apply ρ ← (1 − η)ρ with η = Σδ_K / ∫ρ. Global mass is
   then exact, and signs at S and zero sets are kept.

The exact global M-projection is the same KKT system with an additive scalar ξ
(Cheng–Shen). It is kept as an option for Phase 1 to compare. It needs a
safeguarded semismooth Newton iteration on ξ, nesting the element solves.

### Element solver

Plain semismooth Newton on the element dual degenerates here. S has more
points than the element has modes, and near-zero elements have up to all 61
constraints active (14–20 on average). In the study plain primal-dual active
set cycled on 57–73% of the flagged elements, and regularized variants cycled
or stopped 4% short of the optimum.

The element solver is therefore the Lawson–Hanson active-set method for the
least-distance problem (NNLS):

- **What it is.** A Newton step on the equality-constrained KKT system of each
  active set, with a step-length safeguard and linearly independent active
  sets. It is exact and finite.
- **Near-zero mean.** Elements whose mean is below 1e-8 of their point values
  take the constant mean, the only feasible polynomial.
- **Fallback.** Zhang–Shu scaling is the always-feasible fallback for steps 1
  and 3.
- **Where plain semismooth Newton is used.** Only where the KKT system is
  nondegenerate: the scalar ξ option, and the coupled form if Phase 4 is
  needed.

## Why the order is kept

- **Space.** Point constraints are satisfied by the projected exact solution
  up to O(h^(p+1)), so the feasibility gap does not reduce the order. Measured
  rates: 4.0 at p = 3 and 7.0 at p = 6.
- **Time.** P is the M-orthogonal projection onto a convex set that contains
  the (projected) exact solution up to O(h^(p+1)). It is therefore
  non-expansive:
  ‖P(ρ*) − ρₑ‖_M ≤ ‖ρ* − ρₑ‖_M + O(h^(p+1)).
  The projection cannot amplify the error of the unconstrained step. The
  correction is bounded by the negative part of ρ*, itself bounded by the step
  error wherever the exact density is nonnegative.
  - Measured on recorded states, the correction is 3.4e-3, 5.4e-4 and 1.6e-7
    for the three time steps. That is below the scheme's own dt error of 2.2%
    and 0.75%.
  - The local KKT correction is 1.1–18 times smaller than Zhang–Shu scaling.
- **Two-level history.** BDF2 stability is a two-level (G-norm) argument, and
  the projection acts only on the new level while the drift is extrapolated.
  Cheng and Shen analyse the decoupled form for their schemes; whether that
  analysis covers this extrapolated semi-implicit BDF2 must be checked
  numerically (Phase 2 gate).
- **What is projected.** The projection applies to ρⁿ⁺¹ before the endpoint
  Poisson solve and before the history is committed. The drift, the BDF2
  source and the diagnostics all see the projected density. The transport
  trace is unchanged and stays the next initial guess.

## Cost

Per step at 360,379 triangles, p = 6:

- **Detection.** One batched product of the (N × 28) coefficients with the
  (28 × 61) point table, about 1.2 GFLOP or a few ms on the FP64-limited RTX
  PRO 5000. It shares tables with `ScalarPositivityDiagnostics`, which already
  evaluates volume and lattice points every frame.
- **Element solves.** Only violating elements need one. The counts measured
  (20,000–38,000 at t = 0.5, 119,000 at t = 6.4) are upper bounds, because
  they come from unconstrained trajectories in which violations accumulate.
  The serial Python prototype took 0.12 ms per element (0.45–0.91 s for a
  whole state on 22 CPU workers). The device kernel (one element per warp or
  block, point table and active-set factors in shared memory) is budgeted at
  ≤ 5% of the 0.52 s step.
- **Mass return.** One reduction and one scaling.

## Phases

### Phase 0 — static study (done)

- `scripts/guiding_center/diagnostics/positivity_projection_study.py`
  (`states` and `order` subcommands).
- The [study note](../../research/solver_studies/positivity_projection_static_2026_10_05.md).

### Phase 1 — library projector (implemented)

- **Module.** `hdgfem/transport/positivity.py` (the transport layer may use
  `hdg/cuda`): a `DensityPositivityProjector(space, points=..., backend=...)` with
  `project(field) -> (field, report)`.
  - Host: Numba, parallel over violating elements.
  - Device: CuPy plus a raw kernel, with coefficients kept on the device.
- **Report.** Violating and negative-mean counts, the correction size, η, the
  solver iterations and the time.
- **Shared tables.** Point tables are shared with
  `diagnostics/guiding_center.ScalarPositivityDiagnostics`.
- **Optional modes.** A global-ξ mode and an upper bound (ρ ≤ max ρ⁰, maximum
  principle) are optional variants.
- **Acceptance.**
  - Nonnegativity at S to 10 ulp of the element scale.
  - Mass error ≤ 1e-14 relative.
  - Idempotence, and non-expansiveness on random pairs.
  - Objective ≤ Zhang–Shu, and within 1e-10 of the NNLS reference.
  - Host/device parity.
  - Near-zero and negative-mean elements covered.
  - A small spatial-order regression (p = 3, rate ≥ 3.9 on the study's test
    density).
  - CODEMAP rows updated.
- **Implemented.**
  - The host path is Numba, parallel over the flagged elements. The device
    path is a warp-per-element kernel on the resident coefficients.
  - Device times on stored unprojected states of the positive case
    (360,379 elements, p = 6): 102–119 ms for the 35–38k flagged elements at
    t = 0.5, and 252 ms for the 119k at t = 6.4. Host times: 311–325 ms and
    437 ms. Host and device agree to 2e-16.
  - `tests/test_density_positivity.py` covers:
    - nonnegativity at S and mass on `dub_orth` and Bernstein bases;
    - the NNLS optimum, the Zhang–Shu bound and idempotence;
    - negative-mean elements;
    - the p = 3 spatial rate (4.0 measured; the test requires > 3.8);
    - host/device parity.
- **Open.**
  - Non-expansiveness on random pairs.
  - Point tables shared with the positivity diagnostics.
  - The optional modes.

### Phase 2 — SI-Euler/SI-BDF2 integration and order gates (implemented; manufactured gates passed)

- **Wiring.**
  - Option `density_positivity = "none" | "kkt"`, plus the point set, in the
    guiding-center configuration and in `scripts/reports/record_gpu_showcase.py`.
  - Project the SI-Euler startup step and every BDF2 step, between
    transport and the endpoint Poisson solve.
  - Leave rho_h(0) uncorrected. The initial Poisson and first transport
    solves only see its moments, and the L2 projection makes them the
    moments of the exact nonnegative density. The first correction is the
    post-processing of rho_h(dt).
  - Record the projector report in the per-step metrics.
  - Implemented in the guiding-center runner, the SI steppers and
    `scripts/reports/record_gpu_showcase.py` (`--density-positivity kkt`,
    positive case).
  - Preset: `positive_turbulence_star_si_bdf2_kkt_p6_h005_dt0015625_t6p4_holoviz`.
    It uses the README initial data, a dt of 0.0015625 and Holoviz plots;
    see `scripts/guiding_center/example_runs.md`.
- **Temporal-order gate.**
  - Rigidly rotate a nonnegative profile with quadratic zero contact under a
    prescribed divergence-free velocity, with the reusable transport solver
    and `bdf2_transport_data`.
  - At p = 6 on a fine mesh, the observed dt order with the projection must
    be ≥ 1.9 and its error ≤ 1.1 times the unconstrained error.
  - At small dt, the observed h order must be ≥ p + 0.8.
  - **Passed 2026-10-05**
    ([order gates](../../research/solver_studies/positivity_order_gates_2026_10_05.md),
    `scripts/guiding_center/diagnostics/positivity_order_gates.py`).
  - Time: a rotating bump at p = 6 on 2,048 triangles reaches a dt order of
    1.97 with and without the projection. The KKT error is 0.993–0.9996
    times the unconstrained error.
  - At affordable time steps, BDF2's error on the moving bump hides its h
    order. A radial ring is an exact steady solution of the same rotation, so it
    isolates the spatial error while the projector acts every step. Its h
    orders on the finest mesh pair are 4.33 at p = 3 and 6.83 at p = 6;
    the unconstrained runs reach 4.29 and 6.81.
  - The KKT/unconstrained error ratio there is 1.02 at p = 3 and 1.0002 at
    p = 6, and at most 1.10 on the coarsest meshes.
- **Coupled gate.**
  - Repeat the [three-dt check](../../getting_started/gpu_showcase.md#time-step)
    of the positive case to t = 0.5 with the projection.
  - The density self-convergence order must not fall below the unconstrained
    1.56.
  - Enstrophy loss and energy drift must stay within 1.5 times the
    unconstrained values.
  - The minimum at S must be ≥ −1e-13 · max ρ.
  - **Manufactured coupled gate passed 2026-10-05.**
    - A nonnegative radial vortex translating in a uniform drift is an
      exact guiding-center solution. Its potential is in closed form.
    - Each step uses real HDG Poisson and transport solves, at p = 6 on
      2,048 triangles.
    - The dt order is 1.98 with and without the projection. The KKT error is
      0.997–1.0008 times the unconstrained error.
  - **Open: the full-size positive-case check above.** It needs about
    2,240 GPU steps on 360,379 triangles and is left for the user to run.
    `record_gpu_showcase.py --density-positivity kkt` applies the device
    projector in the showcase loop. Its rows and metadata record the KKT
    reports.
  - Compare the series, and `--pair` each state with the unconstrained
    `chk_positive_global_h005_dt*` checkpoints of the static study:
    ```bash
    python -m scripts.reports.record_gpu_showcase --name chk_positive_kkt_h005_dt0015625 \
      --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
      --cutoff 5 --h 0.005 --dt 0.0015625 --steps 320 --every 32 --seconds 7200 \
      --max-loss 0.9 --poisson-tau global --density-positivity kkt

    python -m scripts.reports.record_gpu_showcase --name chk_positive_kkt_h005_dt00078125 \
      --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
      --cutoff 5 --h 0.005 --dt 0.00078125 --steps 640 --every 32 --seconds 7200 \
      --max-loss 0.9 --poisson-tau global --density-positivity kkt

    python -m scripts.reports.record_gpu_showcase --name chk_positive_kkt_h005_dt000390625 \
      --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
      --cutoff 5 --h 0.005 --dt 0.000390625 --steps 1280 --every 32 --seconds 7200 \
      --max-loss 0.9 --poisson-tau global --density-positivity kkt

    python -m scripts.reports.compare_showcase_states --output positive_kkt_dt.json --series \
      outputs/readme_showcase/chk_positive_kkt_h005_dt0015625.restart.npz \
      outputs/readme_showcase/chk_positive_kkt_h005_dt00078125.restart.npz \
      outputs/readme_showcase/chk_positive_kkt_h005_dt000390625.restart.npz

    # Unconstrained checkpoints of the same check, from the static study.
    python -m scripts.reports.compare_showcase_states --output positive_kkt_vs_none.json \
      --pair <unconstrained>/chk_positive_global_h005_dt0015625.restart.npz outputs/readme_showcase/chk_positive_kkt_h005_dt0015625.restart.npz \
      --pair <unconstrained>/chk_positive_global_h005_dt00078125.restart.npz outputs/readme_showcase/chk_positive_kkt_h005_dt00078125.restart.npz \
      --pair <unconstrained>/chk_positive_global_h005_dt000390625.restart.npz outputs/readme_showcase/chk_positive_kkt_h005_dt000390625.restart.npz
    ```

### Phase 3 — device performance and the dt study

- **Overhead.** Projector overhead ≤ 5% of step time at the showcase size.
  Report per-step violating counts with the projection active.
- **dt study.** Find the largest dt whose t = 0.5 density error (Richardson)
  and conservation diagnostics meet the published accuracy targets, and
  report the wall-time gain over dt = 0.000390625.
- **Long run.** One long positive run (to t = 6.4) at the chosen dt, compared
  with the published run statistically (enstrophy, energy, charge,
  spectra/structure).

### Phase 4 — only if a Phase 2 gate fails

- **Coupled form.** HDG-condensed coupled KKT with semismooth Newton on the
  trace system and an AMGX solve per iteration, if the decoupled projection
  loses order or adds too much dissipation at the target dt.
- **Local mass return.** Neighbour-local mass return instead of the global η,
  if negative-mean elements make the global return visibly nonlocal.
- **Other schemes.** SI-BDF3, IMEX-ARK3 (stage-wise projection) and
  H1/H2-BDF3 follow only after BDF2 passes.

## Risks and open questions

- **Dissipation.** The projection is a nonsmooth correction. At large dt it
  may add numerical dissipation (enstrophy loss) even though the order is
  kept. Phase 3 measures this.
- **Positivity only at S.** Plotted rasters may still show values of order
  1e-3 below zero unless S includes the plotting lattice.
- **Element counts.** Per-step counts with the projection active are unknown.
  The upper bounds above already fit the cost budget.
- **Two-level analysis.** The BDF2 two-level analysis is not covered by the
  non-expansiveness argument alone, hence the Phase 2 gate.

## References

- X. Zhang, C.-W. Shu, J. Comput. Phys. 229 (2010): maximum-principle and
  positivity-preserving scaling limiters for DG.
- J. J. W. van der Vegt, Y. Xia, Y. Xu, SIAM J. Sci. Comput. 41 (2019):
  positivity-preserving KKT limiters for time-implicit DG solved by semismooth
  Newton.
- Q. Cheng, J. Shen, Comput. Methods Appl. Mech. Engrg. 391 (2022): the
  Lagrange-multiplier approach to positivity-preserving schemes, coupled and
  operator-splitting forms.
- M. Hintermüller, K. Ito, K. Kunisch, SIAM J. Optim. 13 (2002): the
  primal-dual active set strategy as a semismooth Newton method.
- C. L. Lawson, R. J. Hanson, *Solving Least Squares Problems* (1974):
  least-distance programming by NNLS.
