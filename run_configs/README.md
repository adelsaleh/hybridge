# Run configurations

HYBRIDGE keeps small inputs for its general solver examples and benchmarks here.

- `adr_unified_l5/`: stationary ADR campaign inventory and its mesh inputs.
- `diff_rea_p6_lc01_solver_benchmarks.json`: dated stationary diffusion solver
  sweep; its identifier is retained for the recorded results.

Store exact arguments and problem parameters, and distinguish measured solver
results from defaults. Large logs and generated fields belong in `run_outputs/`.

Guiding-center research response files and their complete case descriptions live
in the [research repository](../../guiding-center-cg-hdg/run_configs/README.md).
The small HYBRIDGE [BDF2 showcase](../scripts/guiding_center/README.md) takes its
few options directly on the command line.
