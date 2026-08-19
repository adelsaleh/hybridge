# Solver Studies

These documents preserve dated measurements and conclusions. They support
reproducibility but do not define current solver defaults or backend support.

- [`advection_reaction_2026_07.md`](advection_reaction_2026_07.md): upwind-SCC,
  upwind block-GS, Cupyx Krylov/ILU, and AMGX comparisons.
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

Raw reports and machine-readable outputs remain under [`../../../run_logs/`](../../../run_logs/).
