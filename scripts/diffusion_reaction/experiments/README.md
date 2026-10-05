# Diffusion-Reaction Experiments

This directory contains research solvers and specialized kernels that are not
installed with `hybridge` and are not part of the supported solver API.

- `bootstrap_initial_guess.py` solves the same mesh at a lower polynomial order,
  elevates the trace, and uses it as an initial guess for the target solve.
- `test7_fused.py`, `test7_fused_backend.py`, and `test7_fused_kernels.py`
  implement the hard-coded tensor-diffusion Test 7 fused Numba experiment. Its
  focused parity test is `tests/test_diffusion_reaction_test7_fused_experiment.py`.

- `tensor_schur.py` compares in-kernel TF32 Schur products with the original
  FP64 and FP32 fused assembly kernels; see the
  [measured study](../../../docs/research/solver_studies/tensor_schur_prototype_2026_09_25.md).
