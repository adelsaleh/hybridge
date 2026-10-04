# Solver Studies

These documents preserve dated measurements and conclusions. They support
reproducibility but do not define current solver defaults or backend support.

- [Advection-reaction AMGX configuration findings](adv_rea_amgx_config_2026_07_20.md) and
  [raw CUDA fused cooperative LU findings](raw_cuda_fused_coop_lu_2026_07_20.md):
  July 2026 p=6--8 AMGX configuration, tolerance and modal-trace sweeps, and
  fused raw-CUDA matrix-level parity and reconstruction checks.

- [Advection assembly baseline, RTX PRO 5000 Blackwell](advection_assembly_baseline_2026_10_03.md):
  native fused/split3, hybrid and pure-library (cuTENSOR/cuBLAS/MAGMA)
  assembly for p=4--9 in FP64 and FP32, stage-2 batched LU alone, the machine
  and software stack, predictions for FP64-capable GPUs and H100 steps.

- [Raw CUDA ADR tensor assembly and reconstruction](raw_cuda_adr_tensor_2026_09_28.md):
  FP64 parity, native stationary CSR/BSR solves and matched assembly timings.

- [Numba ADR tensor specialization](numba_adr_tensor_2026_09_25.md):
  variable scalar/tensor parity, manufactured convergence and 42 warmed
  comparisons with coupled LU.

- [Host Numba diffusion Schur caches](numba_diffusion_schur_2026_09_25.md):
  LU/Cholesky parity, cache invalidation and 108 warmed phase/thread benchmarks.

- [Repeated ITER Poisson solves](iter_repeated_poisson_2026_09_24.md):
  matched fixed-source p=6 measurements, a faster native multigrid policy,
  and dense-inverse/sparse-LU feasibility checks.

- [Temporary SciPy LU to GPU experiment](iter_scipy_lu_gpu_experiment.md):
  saved explicit factors, retained SpSV/SpSM analysis, CUDA graphs, and live
  output; GPU LU wins on 2,095 ITER triangles, while pMG-AMG wins on 80,843
  triangles at p=6 despite the factors fitting comfortably in device memory.

- [hp-AMG hierarchy choices and literature](../../algorithms/hp_amg/hierarchies.md):
  the Poisson design review is maintained with the shared hp-AMG solver
  formalism; covers transfers, smoothers, h/p levels, and BSR mapping.

- [General nonsymmetric pMG–AMG design](nonsymmetric_pmg_design_2026_09_22.md):
  proposed block Petrov–Galerkin hierarchy, AIR references, and small-matrix
  algebra checks; no production implementation or HDG benchmark claim.

- [Unified ADR L1–L5 campaign](adr_unified_l5_campaign.md): portable iterative
  inventory, approximately 400k-triangle L5 extensions, memory guards and resume;
  explicit CUDA-12/legacy-BSR instructions for AMU V100 nodes. Cluster builds
  and numerical validation remain user-run.

- [Closed-loop ADR stress runner](adr_closed_loop_stress_runner.md): prepared
  runner for the nine-lobed trapping, crossing and orthogonal-transport cases;
  planning, launch commands and validation limits. No stress solver results yet.

- [Combined ADR scaling results](adr_scaling_2026_09_17/README.md): one insertable LaTeX section with smooth and oscillatory problems, three diffusion classes, square and five-lobed annular h/p sweeps, ASM+PP, AMGX BSR FGMRES/BiCGSTAB and native hp-BSR. The oscillatory subsection preserves the initial 22k stress results and adds scaling to 100k triangles.

- [`adr_solver_comparison_2026_09_17.md`](adr_solver_comparison_2026_09_17.md): matched stationary ADR at 99,458 triangles/p=6, ASM+PP versus AMGX BSR and native hp-BSR; recorded tuning, physical residuals, application counts/timings, memory, and tolerance-triggered restart diagnosis.

- [`hybrid_hierarchy_bsr_feasibility_2026_09.md`](hybrid_hierarchy_bsr_feasibility_2026_09.md): lossless hybrid hierarchy exports and saved p=6 CSR/BSR products; full BSR projects slower on both meshes, with small selective opportunities.

- [`asm_pp_bsr_opportunities_2026_09.md`](asm_pp_bsr_opportunities_2026_09.md): GMRES branch review, assembled BSR ASM helpers, and 150k/300k finest-smoother tests: the measured ASM variants do not close the hybrid iteration gap.

- [`guiding_center_poisson_timings_2026_09.md`](guiding_center_poisson_timings_2026_09.md): saved SI Euler, BDF2, H1/H2-BDF3 and IMEX-ARK3 timings, coarse AMGX attribution, and optimization candidates.

- [`advection_reaction_2026_07.md`](advection_reaction_2026_07.md): upwind-SCC,
  upwind block-GS, Cupyx Krylov/ILU, and AMGX comparisons.
- [Discontinuous advection trace diagnosis](discontinuous_advection_trace_2026_09_25.md):
  the original double-outflow fixture, positive-reaction and boundary controls,
  and verification of the existing averaging policy in all four assembly backends.
- [`diffusion_assembly_2026_07.md`](diffusion_assembly_2026_07.md): NumPy
  materialization versus fused Numba host assembly timing and memory study.
- [`diffusion_amgx_2026_07.md`](diffusion_amgx_2026_07.md): modal-trace matrix
  scaling, hierarchy, and preconditioner investigation.
- [`diffusion_amgx_tolerance_2026_08.md`](diffusion_amgx_tolerance_2026_08.md):
  bounded practical-tolerance recheck of the modal BICGSTAB/classical-AMG
  fallback.
- [`guiding_center_host_device_2026_08.md`](guiding_center_host_device_2026_08.md):
  version-pinned SciPy ILU/upwind-SCC, PARDISO, raw-CUDA/AMGX, and strict
  true-residual GMRES findings for the Gaussian-annulus k=3 case.
- [`classical_amg_bsr_2026_08.md`](classical_amg_bsr_2026_08.md): validated
  hybrid classical-AMG hierarchy with a BSR fine operator, including a
  radius-5 Poisson sweep over 99,896–150,209 triangles and degrees 1–6;
  machine-readable samples are in
  [`classical_amg_bsr_sweep_samples_2026_08.csv`](classical_amg_bsr_sweep_samples_2026_08.csv).
- [`amgx_vs_face_dense_2026_08.md`](amgx_vs_face_dense_2026_08.md):
  matched radius-5 Poisson comparison of AMGX CSR against face-dense GMRES
  with fused ASM and polynomial preconditioning.
- [`face_dense_primitives_2026_08.md`](face_dense_primitives_2026_08.md):
  operator, ASM, and polynomial-application attribution for that face-dense
  solver; machine-readable samples are stored beside the report.

- [`guiding_center_temporal_cfl_2026_09.md`](guiding_center_temporal_cfl_2026_09.md):
  SI Euler, BDF2 and predictor–corrector manufactured temporal convergence and
  star-with-hole vortex-gas CFL, enstrophy and HDG H1 diagnostics;
  [machine-readable samples](guiding_center_temporal_cfl_samples_2026_09.csv).

Raw reports and machine-readable outputs remain under `run_logs/` (local, untracked).

Guiding-center study artifacts are under
`run_outputs/guiding_center/convergence/` (local, untracked).

- [`diocotron_ark3_2026_09.md`](diocotron_ark3_2026_09.md): smooth-first disk
  instability, radial eigenvalue references, every-stage positivity, invariants,
  bounded IMEX-ARK3 screening and provisional high-mode resolution ladder.

- [`diocotron_high_modes_2026_09.md`](diocotron_high_modes_2026_09.md): m=64/128
  Gaussian annuli, analytical/smooth spectra, growth screening and control limitations.

- [Raw assembly shared-memory investigation](raw_assembly_shared_memory_2026_09_25.md): capture commands, bounded timings, and outstanding hardware-counter evidence.

- [Fused tensor Schur and FP32 assembly comparison](tensor_schur_prototype_2026_09_25.md): degree-dependent block sweeps, accuracy checks, and instruction evidence.
