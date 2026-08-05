# Diffusion-Reaction Experiments

This directory contains research solvers and specialized kernels that are not
installed with `hdgfem` and are not part of the supported solver API.

- `bootstrap_initial_guess.py` solves the same mesh at a lower polynomial order,
  elevates the trace, and uses it as an initial guess for the target solve.
- `test7_fused.py`, `test7_fused_backend.py`, and `test7_fused_kernels.py`
  implement the hard-coded tensor-diffusion Test 7 fused Numba experiment. Its
  focused parity test is `tests/test_diffusion_reaction_test7_fused_experiment.py`.
