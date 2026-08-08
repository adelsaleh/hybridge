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

Raw reports and machine-readable outputs remain under [`../../../run_logs/`](../../../run_logs/).
