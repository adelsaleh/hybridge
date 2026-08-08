# Modal Diffusion AMGX Practical-Tolerance Recheck: August 2026

Status: bounded historical performance check, not a solver recommendation.

## Question

The July study left the modal `BICGSTAB + classical AMG` fallback open for a
recheck at practical requested tolerances. Two single-sample runs tested
`1e-9` and `1e-10` on the existing heavy p6 case.

## Common Setup

- trigonometric Poisson case on a disc mesh;
- polynomial order 6, mesh size 0.04, and `dub_orth` element basis;
- `legendre-modal` trace basis and symmetric volume quadrature;
- raw-CUDA direct CSR assembly with block size 128;
- `BICGSTAB` with
  `configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json`;
- maximum 2,000 AMGX iterations;
- NVIDIA Quadro RTX 6000, with each row representing one run.

The reduced system had 113,878 triangles, 1,192,968 trace unknowns, and
41,676,852 nonzeros.

## Results

| Requested tolerance | Iterations | Setup | Solve | Physical residual | L2 error | Total measured |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-9 | 2,000 | 1.403 s | 5.772 s | 6.564e-9 | 3.090e-5 | 15.282 s |
| 1e-10 | 2,000 | 0.381 s | 5.063 s | 1.470e-9 | 3.796e-5 | 8.758 s |

Both runs reached the iteration cap without attaining the requested AMGX
residual. The sweep wrapper reported successful execution because the solves
completed and produced finite solutions; that status does not mean the
requested solver tolerance was reached.

## Conclusion

Lowering the requested tolerance did not make this fallback meet its internal
convergence target on the heavy modal case. Keep it as a historical fallback
only; these samples do not justify a config or default change. The timing and
error variation between two single runs is not a performance comparison.

## Reproducibility Evidence

Commands:

```text
.venv/bin/python -m scripts.gpu.sweep_diffusion_amgx_preconditioners -o 6 -ms 0.04 --amgx-tolerance 1e-9 --only-variants modal_bicgstab_classical_control --no-include-generated --timeout 900 -v 1
.venv/bin/python -m scripts.gpu.sweep_diffusion_amgx_preconditioners -o 6 -ms 0.04 --amgx-tolerance 1e-10 --only-variants modal_bicgstab_classical_control --no-include-generated --timeout 900 -v 1
```

Raw logs:

- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260807_173736.csv`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260807_173736.jsonl`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260807_173811.csv`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260807_173811.jsonl`

The broader modal investigation is in
[`diffusion_amgx_2026_07.md`](diffusion_amgx_2026_07.md).
