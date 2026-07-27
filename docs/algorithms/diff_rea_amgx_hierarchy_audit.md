# Diffusion-Reaction AMGX Hierarchy Audit

Date: 2026-07-26

This note records the AMGX hierarchy-stat audit for the p6 diffusion-reaction GPU runner after diagonal scaling and non-Chebyshev preconditioner sweeps did not explain the slow `dub_orth + legendre-modal` modal trace solve.

The diagnostic driver is:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib .venv/bin/python -m scripts.gpu.inspect_diff_rea_amgx_hierarchy
```

It writes a temporary copy of the selected AMGX config with `print_grid_stats=1`, `print_solve_stats=1`, and AMGX timing/history output enabled, then runs `scripts.gpu.run_diff_rea_gpu4_hdg`.  It stores parsed summaries as CSV/JSONL under `run_logs/` and keeps the raw runner/AMGX output for each case.

## Baseline Aggressive Chebyshev/L1 Hierarchy

Config: `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`

Run log: `run_logs/diff_rea_amgx_hierarchy_o6_ms0p18_ms0p04_20260726_223440.csv`

Common options:

- `case=trigonometric-poisson`
- `mesh_type=disc`
- `order=6`
- `basis=dub_orth`
- `volume_quadrature=symmetric`
- `assembly_backend=raw-cuda`
- `raw_matrix_format=csr`
- `raw_block_size=128`
- `amgx_solver=PCGF`
- `amgx_tolerance=1e-12`

| mesh | trace basis | status | levels | coarse rows | operator complexity | iterations | AMGX setup | AMGX solve | physical residual |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | legacy-lagrange | ok | 2 | 3,738 | 1.02278 | 31 | 0.2576s | 0.0875s | 5.271e-13 |
| 0.18 | legendre-modal | failed | - | - | - | - | - | - | AMGX setup OOM |
| 0.04 | legacy-lagrange | ok | 4 | 5,601 | 1.03785 | 32 | 0.4414s | 0.4494s | 7.503e-13 |
| 0.04 | legendre-modal | ok | 4 | 2,308 | 1.00762 | 1,923 | 0.2246s | 23.8678s | 9.514e-13 |

Fine p6 level rows/NNZ:

| trace basis | L0 rows/NNZ | L1 rows/NNZ | L2 rows/NNZ | L3 rows/NNZ |
|---|---:|---:|---:|---:|
| legacy-lagrange | 1,192,968 / 41,676,852 | 74,686 / 949,800 | 25,476 / 480,646 | 5,601 / 147,117 |
| legendre-modal | 1,192,968 / 41,676,852 | 22,297 / 212,901 | 7,271 / 78,803 | 2,308 / 26,030 |

The modal matrix has the same fine-grid size and NNZ as nodal, but AMGX builds a much smaller hierarchy for the modal coordinates.  This lowers setup and operator complexity, but convergence collapses: average AMGX convergence rate is about `0.9857` for the fine modal run versus about `0.4179` for the fine nodal run.  The issue is therefore not excessive modal hierarchy memory or raw-CUDA matrix asymmetry; it is the AMG coarse representation/smoother effectiveness for modal trace coordinates.

## Coarse Modal Setup Failure

The p6/ms0.18 modal aggressive config fails during AMGX setup with an out-of-memory path in `DenseLUSolver::solver_setup`, followed by a PyAMGX `CUDA kernel launch error`.

Raw output:

```text
run_logs/diff_rea_amgx_hierarchy_o6_ms0p18_legendre_modal_20260726_223440.out
```

A diagnostic-only config with `coarse_solver=NOSOLVER` was used to expose the hierarchy that AMGX would otherwise hide behind the DenseLU failure.  It coarsened the modal matrix as:

| level | rows | NNZ |
|---:|---:|---:|
| 0 | 59,227 | 2,055,795 |
| 1 | 1,130 | 10,438 |
| 2 | 359 | 3,551 |
| 3 | 113 | 1,079 |
| 4 | 33 | 261 |
| 5 | 8 | 38 |
| 6 | 2 | 4 |

AMGX source audit: `dense_lu_num_rows` is registered as the threshold where DenseLU is triggered, and in `amg.cu` it replaces `min_coarse_rows` when DenseLU is selected.  The project config currently sets `dense_lu_num_rows=2048`, so the modal hierarchy can stop around the first aggressive coarse level instead of continuing to a very small coarse grid.

## Lower DenseLU Trigger Diagnostic

Temporary config: `/tmp/hdgfem/amgx_sweeps/diff_rea_pcgf_cheb_l1_aggressive_dense128.json`

Change relative to the project config:

```json
{
  "dense_lu_num_rows": 128,
  "dense_lu_max_rows": 512
}
```

Run log: `run_logs/diff_rea_amgx_hierarchy_o6_ms0p18_ms0p04_20260726_223752.csv`

| mesh | trace basis | status | levels | coarse rows | operator complexity | iterations | AMGX setup | AMGX solve | physical residual |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | legacy-lagrange | ok | 4 | 268 | 1.03679 | 32 | 0.1379s | 0.0547s | 5.188e-13 |
| 0.18 | legendre-modal | ok | 3 | 359 | 1.00680 | 370 | 0.1325s | 0.4093s | 8.940e-13 |
| 0.04 | legacy-lagrange | ok | 6 | 173 | 1.03867 | 32 | 0.1804s | 0.4071s | 8.466e-13 |
| 0.04 | legendre-modal | ok | 6 | 214 | 1.00785 | 1,922 | 0.1840s | 23.1004s | 9.570e-13 |

Production-style no-grid-stat timing checks on the fine p6/ms0.04 case gave:

| trace basis | iterations | AMGX setup | AMGX solve | physical residual |
|---|---:|---:|---:|---:|
| legacy-lagrange | 32 | 0.1785s | 0.4074s | 8.466e-13 |
| legendre-modal | 1,922 | 0.1824s | 23.2439s | 9.570e-13 |

The lower DenseLU threshold fixes the modal p6/ms0.18 setup failure and lowers nodal setup cost in these samples.  It does not fix the modal fine-grid iteration problem.

## Interpretation

- The modal aggressive hierarchy is smaller, not larger, than the nodal hierarchy.  Lower operator complexity is not sufficient for good PCGF convergence.
- The poor modal PCGF behavior is consistent with AMG interpolation/coarse spaces not representing high-order modal edge error components well enough after aggressive coarsening.
- Scalar diagonal scaling is not enough because it leaves the modal coordinate/coarse-space issue intact.
- The current DenseLU trigger is also a robustness issue.  `dense_lu_num_rows=2048` can stop modal aggressive coarsening early and trigger a DenseLU setup failure on the p6/ms0.18 case.  A smaller trigger such as `128` avoids that failure and keeps nodal iteration counts unchanged in the checked cases.

## Current Recommendation

Do not change the production AMGX config from a single diagnostic run.  The lower DenseLU threshold is promising and should be repeated across p4-p8, at least `ms=0.18`, `ms=0.08`, and `ms=0.04`, before promotion.

For modal performance, keep focusing on basis-aware modal trace scaling or mass-normalized modal coordinates.  The hierarchy audit supports the earlier conclusion that the raw-CUDA modal matrix is not the root issue; AMGX is building an inexpensive hierarchy that is ineffective for modal PCGF convergence.
