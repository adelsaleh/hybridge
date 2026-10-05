# Native Face-Block hp-Multigrid Poisson

The [hp-AMG solver family and formalism](../algorithms/hp_amg/README.md)
derive this native pMG-AMG variant's modal hierarchy, V-cycle, PCGF recurrence,
and SPD assumptions alongside alternative geometric and algebraic hierarchies.

`solver="fb-hp-mg-pcg"` selects HYBRIDGE's reusable native Poisson backend for supported raw-CUDA problems at polynomial degrees 4 through 6. The condensed trace operator is assembled directly as face BSR in the Legendre-modal assembly basis. Setup applies the one-time congruence transformation to orthonormal modal coordinates, builds a direct `p -> 0` hierarchy, and retains the operator, dense face-block diagonal inverses, Chebyshev spectral estimates, Krylov/V-cycle workspaces, and one fixed scalar-AMGX hierarchy at `p=0`.

The production cycle uses order-2 Chebyshev smoothing with symmetric `1+1` pre/post smoothing. Generic cuSPARSE BSR descriptors own ordinary finest-level matrix-vector products; HYBRIDGE raw-CUDA kernels fuse BSR traversal, residual formation, dense block-Jacobi inversion, and Chebyshev updates inside the smoother. PCG checks positive curvature and periodically refreshes the FP64 true residual. Residual norm, A-curvature, and preconditioned curvature are downloaded together, so the hot loop has one host synchronization per iteration instead of three. Warm trace guesses enter in the assembly basis and converged traces are transformed back before normal HDG field/flux reconstruction.

The native outer iteration now uses the flexible PCGF beta update already used
by the diagnostic prototype; the public `fb-hp-mg-pcg` selector is unchanged.
It retains one best true-residual iterate, restarts the direction when a true
refresh differs from the recursive residual by more than 10% of the true norm,
and ends a stalled attempt after six true checks without 1% improvement in the
best norm. These are recovery safeguards, not relaxed convergence: the original
assembled matrix residual must still meet the requested target. The robust
guiding-center policy uses `p -> floor(p/2) -> ... -> 0`, order-4 Chebyshev,
2+2 pre/post sweeps, and a stronger fixed scalar coarse cycle. Standard policy
retains the original preconditioner tuning described above.

The `fast` policy uses direct `p -> 0`, order-1 Chebyshev and balanced 1+1
fine/coarse sweeps. The p=6 ITER positive-turbulence preset selects it based on
fixed-operator diagnostics; other presets keep their existing policy, including
the p=5 ITER recovered-field preset. Select a policy explicitly with
`--poisson-fb-hp-mg-preconditioner-policy fast` (or `standard`, `robust`).
All policies retain the same original-matrix residual acceptance and fallback
checks. See the [ITER repeated-solve study](../research/solver_studies/iter_repeated_poisson_2026_09_24.md).

The scalar coarse application explicitly requests `fixed_amg_cycles=1` from
the shared AMGX wrapper. Inner residual stopping and history collection are
disabled; outer PCGF still performs its normal convergence checks. This both
avoids unnecessary coarse reductions and preserves fixed work for arbitrarily
small nonzero RHS vectors.

Standalone fixed-work tuning is available through
`FaceBlockHpMgPcgSolver(..., preconditioner_policy="standard",
preconditioner_tuning={"chebyshev_order": 3})`. The shared policy resolver
also accepts `sweeps`, `coarse_sweeps`, and `coarse_cycle` (`V` or `W`).
Sweep counts always remain balanced; the hierarchy still applies exactly one
coarse cycle and retains its symmetry/positive-curvature gates. Omitted tuning
leaves production defaults unchanged. The resolved settings are exposed as
`preconditioner_parameters` for reproducible benchmark records.

In the diffusion solve, every native true check uses the original assembled
matrix action, not the separately stored orthonormal congruence. Its norm is
measured before mapping the residual back to Krylov coordinates. Thus a small
roundoff discrepancy at the target triggers further native PCGF iterations
from the current trace, rather than a full AMGX fallback. The same original
residual ranks the single best checkpoint; an independent final check remains
mandatory. Standalone callers can provide this action through
`FaceBlockHpMgPcgSolver.solve(..., assembly_matvec=...)`.

At runner verbosity `-v 3`, native PCGF prints and flushes each iteration as it
finishes, rather than replaying the table after the solve. Rows distinguish
recursive residuals from true-residual checks. The scalar `p=0` AMGX cycle
remains quiet inside each preconditioner application; the final summary reports
its count/time, residual restarts, and best/terminal residuals separately.
AMGX itself uses a flushed native print callback, including with terminal-log
redirection, so its progress does not wait in a C stdout buffer.

The reusable diffusion solver keys the hierarchy to the fixed operator, not the RHS or Krylov tolerance. Source and boundary changes therefore refresh only RHS/reconstruction data. The warm compact Schur-Cholesky path retains ``S_e^-1 B_e`` plus one scalar source solution, fuses source condensation/face scatter, and reconstructs the mixed field with a raw-CUDA response kernel; it does not rebuild BSR structure or upload matrix values. `poisson_time_operator_assembly` and `poisson_time_rhs_assembly` separate cold and repeated work. `solve.fb_hp_mg.setup_outer`, `solve.fb_hp_mg.krylov`, hierarchy reuse, workspace bytes, symmetry/curvature diagnostics, and fallback state are recorded separately. The cold solve headline includes native hierarchy construction, while `global_solve_result.solve_elapsed_seconds` remains Krylov-only. If setup or a solve fails a runtime, symmetry, curvature, or convergence gate, the solver builds the established fine-BSR/scalar-AMGX hybrid once and reuses that fallback for subsequent RHS solves.

The scalar p=0 AMGX defaults use strength threshold `0.40` and dense-LU thresholds `128/256`, with one fixed classical V-cycle and symmetric 1+1 L1-Jacobi smoothing. These three parameter changes reduced complete Poisson time by 11.1–14.0% in the 157,280-triangle p=4–6 Euler vortex-gas validation. The [parameter-tuning study](../development/plans/face_block_hp_multigrid.md#scalar-p0-amgx-parameter-tuning-2026-09-13) records actual level sizes, GPU costs, correctness checks, rejected settings, and the scope of that result.

## Guiding-center policy

The device profile selects native Poisson for p=4--6 with raw-CUDA BSR, a Legendre-modal Poisson trace, no Poisson scaling, and cached Schur-Cholesky local factors. Lower orders and unsupported configurations retain the hybrid AMGX path. Transport independently retains its measured trace basis and uses scaled tangent-boundary raw-CUDA BSR BICGSTAB with the accepted density trace as its initial guess. Timing step 0 records the actual cold solve as `first_poisson_wall_time`; when an equilibrium solve precedes the perturbed initial state, `equilibrium_poisson_wall_time` and `initial_poisson_wall_time` preserve both phases and `poisson_time_total` includes both.

Two matched six-step benchmark presets are available:

- `diocotron_gaussian_annulus_k3_p6_150k_fb_hp_mg_6step`
- `diocotron_gaussian_annulus_k3_p6_150k_hybrid_amgx_6step`

Both request Gaussian-annulus k=3, p=6, mesh size `0.0068`, at least 150,000 triangles, SI-Euler, six accepted steps, per-step diagnostics, and no plotting. Treat step 1 as warm-up when comparing steps 2--6. These are expensive production benchmarks and are not part of the normal test suite.

## Periodic electric-field postprocessing

`poisson_flux_postprocess_every=N` reconstructs the accepted Poisson flux every N accepted steps. `poisson_flux_postprocess_space="RT_projection"` selects the unisolvent Raviart--Thomas `RT_p` recovery and `poisson_postprocessing_backend="raw-cuda"` selects the raw-CUDA per-element moment assembler/solver. The result exposes the degree-p+1 field as `postprocessed_flux`; diagnostics record its norm separately as `q_l2_postprocessed`, while conservation histories continue to use the standard HDG flux so the cadence does not introduce artificial jumps.

The pure advection solver currently requires its velocity field in the transport DG space. Consequently periodic RT output is retained for higher-accuracy user diagnostics and output, while time integration continues with the standard degree-p Poisson flux. Cross-space device-resident advection coefficients are tracked in `TODO.md`.

Implementation entry points are `hybridge/linalg/multigrid/face_hp.py`, `hybridge/linalg/multigrid/policy.py`, `hybridge/linalg/gpu/legendre_face_bsr.py`, `hybridge/mixed/postprocess/rt_raw_cuda.py`, and `hybridge/solvers/diffusion_reaction.py`. Retained performance evidence and rejected alternatives remain in the development plan.

## Guiding-center diagnostics cadence

The `diocotron_k50_p6_150k_raw_cuda_bsr_plot30` production preset writes accepted-state diagnostics every 30 steps, aligned with plotting (plus the initial and final states). Mass, field extrema, flux norms, equilibrium differences, and azimuthal harmonics reduce resident DG coefficients on CUDA and transfer one compact scalar vector. `diagnostics_backend="cuda"` and `diagnostics_device_reduction_time` make this observable in JSONL output; plotting remains the only cadence-driven full-field host consumer.
