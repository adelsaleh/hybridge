# Native Face-Block hp-Multigrid Poisson

The [hp-AMG solver family and formalism](../algorithms/hp_amg/README.md)
derive this native pMG-AMG variant's modal hierarchy, V-cycle, PCGF recurrence,
and SPD assumptions alongside alternative geometric and algebraic hierarchies.

`solver="fb-hp-mg-pcg"` selects HYBRIDGE's reusable native Poisson backend for supported raw-CUDA problems at polynomial degrees 1 through 9 (`FB_HP_MG_DEGREES` in `hybridge/linalg/multigrid/policy.py`: the fused smoother and raw SpMV kernels are compiled for face block sizes 1..10, and p = 0 has nothing to coarsen). Its recorded performance evidence covers p = 4..6. Research degree sweeps are maintained in the experiment repository (`../../../guiding-center-cg-hdg/docs/backends/face_hp_mg_pcg.md`). The condensed trace operator is assembled directly as face BSR in the Legendre-modal assembly basis. Setup applies the one-time congruence transformation to orthonormal modal coordinates, builds a direct `p -> 0` hierarchy, and retains the operator, dense face-block diagonal inverses, Chebyshev spectral estimates, Krylov/V-cycle workspaces, and one fixed scalar-AMGX hierarchy at `p=0`.

The production cycle uses order-2 Chebyshev smoothing with symmetric `1+1` pre/post smoothing. Generic cuSPARSE BSR descriptors own ordinary finest-level matrix-vector products; HYBRIDGE raw-CUDA kernels fuse BSR traversal, residual formation, dense block-Jacobi inversion, and Chebyshev updates inside the smoother. PCG checks positive curvature and periodically refreshes the FP64 true residual. Residual norm, A-curvature, and preconditioned curvature are downloaded together, so the hot loop has one host synchronization per iteration instead of three. Warm trace guesses enter in the assembly basis and converged traces are transformed back before normal HDG field/flux reconstruction.

The native outer iteration now uses the flexible PCGF beta update already used
by the diagnostic prototype; the public `fb-hp-mg-pcg` selector is unchanged.
It retains one best true-residual iterate, restarts the direction when a true
refresh differs from the recursive residual by more than 10% of the true norm,
and ends a stalled attempt after six true checks without 1% improvement in the
best norm. These are recovery safeguards, not relaxed convergence: the original
assembled matrix residual must still meet the requested target. The robust
preconditioner policy uses `p -> floor(p/2) -> ... -> 0`, order-4 Chebyshev,
2+2 pre/post sweeps, and a stronger fixed scalar coarse cycle. Standard policy
retains the original preconditioner tuning described above.

The `fast` policy uses direct `p -> 0`, order-1 Chebyshev and balanced 1+1
fine/coarse sweeps. Select `preconditioner_policy="fast"`, `"standard"`, or
`"robust"` on `FaceBlockHpMgPcgSolver`. All policies retain the same
original-matrix residual acceptance and fallback checks. Dated application
measurements remain in the research study (`../../../guiding-center-cg-hdg/docs/research/solver_studies/iter_repeated_poisson_2026_09_24.md`).

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

The scalar p=0 AMGX defaults use strength threshold `0.40`, dense-LU
thresholds `128/256`, one fixed classical V-cycle, and symmetric 1+1 L1-Jacobi
smoothing. The parameter-tuning evidence (`../../../guiding-center-cg-hdg/docs/development/plans/face_block_hp_multigrid.md`)
records the application, measured improvement, rejected settings, and limits.

The general flux-recovery API is documented in the
[manual](../../MANUAL.md#diffusion-reaction-solve-and-postprocessing).
Application-specific recovery cadence and time-integration choices belong to
the experiment drivers.
