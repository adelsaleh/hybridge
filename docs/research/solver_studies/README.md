# Solver Studies

These documents preserve dated measurements and conclusions. They support
reproducibility but do not define current solver defaults or backend support.

- [`advection_reaction_2026_07.md`](advection_reaction_2026_07.md): upwind-SCC,
  upwind block-GS, Cupyx Krylov/ILU, and AMGX comparisons.
- [`diffusion_assembly_2026_07.md`](diffusion_assembly_2026_07.md): NumPy
  materialization versus fused Numba host assembly timing and memory study.
- [`diffusion_amgx_2026_07.md`](diffusion_amgx_2026_07.md): modal-trace matrix
  scaling, hierarchy, and preconditioner investigation.

Raw reports and machine-readable outputs remain under [`../../../run_logs/`](../../../run_logs/).
