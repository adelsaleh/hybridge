# Magnetron study

The crossed-field vane magnetron in the guiding-center limit (D7): the p-ladder and the mesh and
time-step ladders of the vane case, the collisional parallel-vane discharge (article Section 11),
the KKT positivity showcase, and the CG-versus-HDG Poisson timing on the D7 vane meshes (E9).
The current production campaign is `magnetron_d7_production_v2` (V2 compatible drift at p_phi = p,
curved walls refined, SI-BDF2, T = 40, two pulses).

| Script | Role |
|---|---|
| `run_magnetron.py` | Driver of the magnetron campaigns (`--anode-geometry vanes`, emission, Hall parameter, waves, meshes, positivity, snapshots and movies); `--dry-run` builds only the meshes. Starts solves otherwise. |
| `run_poisson_cg_vs_hdg.py` | E9: Poisson time per solve, HDG(p) against condensed CG(p + 1), on the D7 meshes. Starts solves. |
| `magnetron_convergence.py` | Convergence across mesh and time-step ladders on one pixel grid. No solve. |
| `magnetron_order_efficiency.py` | Accuracy per unknown of several degrees against one reference run. No solve. |
| `write_magnetron_tables.py` | LaTeX tables from the efficiency JSON files. No solve. |
| `plot_magnetron.py`, `plot_discharge.py` | Article figures (currents, snapshots, the discharge overview with the SUPG comparison). No solve. |

## Outputs

| Campaign | Directory | Status |
|---|---|---|
| D7 production v2 (ladder, reference, time steps, re-timing) | `run_outputs/article/magnetron_d7_production_v2` | complete (`all-done`, `retime-done`); `campaigns/gc_d7_magnetron` |
| D7 production (first version), pilot, diagnosis, SUPG counterpart | `run_outputs/article/magnetron_d7_production`, `magnetron_d7_pilot`, `magnetron_d7_diagnosis_20261016`, `magnetron_d7_supg` | complete; `campaigns/gc_d7_magnetron` |
| Earlier mesh and time-step convergence shells | `run_outputs/article/magnetron` | complete; `campaigns/gc_d7_magnetron` |
| E9 CG versus HDG, and its exclusive-GPU re-timing | `run_outputs/article/e9_cg_vs_hdg`, `e9_retime_20261015` | complete (`all-done`); `campaigns/gc_e9_cg_vs_hdg` |

## Commands

Options only:

```bash
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_magnetron --help
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg --help
```

One D7 v2 run at p = 6, scale 1 (the launcher's options; `--dry-run` builds the meshes and prints the
configuration, without it the run starts):

```bash
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_magnetron \
  --anode-geometry vanes --vane-width 0.26 --wave-amplitudes 0.05 --wave-speeds 0.25 --hub-edge 0.025 \
  --emission-top 0.6 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time 40 \
  --hall-parameter 30 --electron-temperature 0.01 --emission-period 20 --emission-phase 5 \
  --density-backend device --density-positivity-points quadrature+lattice+dense --velocities compatible \
  --poisson-order-offset 0 --curvature-points 3000 --density-rtol 1e-10 \
  --snapshot-times 10,20,30,40 --prefix d7v2 --verbosity 1 \
  --orders 6 --dts 0.05 --mesh-size 0.03 --band-mesh-size 0.015 --corner-mesh-size 0.001 --cathode-mesh-size 0.003 \
  --density-positivity none --movie-interval 0 --output-dir run_outputs/article/magnetron_d7_production_v2/dry/p6_s1 --dry-run
```

Mesh scales divide the four mesh sizes (0.88, 0.45, 0.227 at p = 2; 1.414, 0.72, 0.36 at p = 4; 2, 1, 0.5
at p = 6); the time-step lane keeps p = 6 at scale 1 with `--dts 0.025`, `0.0125`, `0.00625`; the KKT
showcase adds `--density-positivity kkt`.

E9 (exclusive GPU for the timing; the launcher takes the GPU lock):

```bash
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg --orders 6,7 --output-dir run_outputs/article/e9_cg_vs_hdg
```

Analysis, tables and figures (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.magnetron.magnetron_order_efficiency --reference run_outputs/article/magnetron_d7_production_v2/runs/ref_p6_s0p36/d7v2.jsonl --runs run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl run_outputs/article/magnetron_d7_production_v2/runs/p4_s0p72/d7v2.jsonl --stem d7_ladder --times 10,20,30,40
.venv/bin/python -m scripts.guiding_center.studies.magnetron.magnetron_convergence --runs run_outputs/article/magnetron_d7_production_v2/runs/p6_s2/d7v2.jsonl run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl --parameter mesh --label 'mesh scale' --stem d7_mesh --times 10,20,30,40
.venv/bin/python -m scripts.guiding_center.studies.magnetron.write_magnetron_tables --ladder d7_ladder.json --dt d7_dt.json --ladder-table ../gc-hdg-article/tables/magnetron_ladder.tex --dt-table ../gc-hdg-article/tables/magnetron_dt.tex
.venv/bin/python -m scripts.guiding_center.studies.magnetron.plot_magnetron --runs run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl --stem magnetron --times 10,20,30,40
.venv/bin/python -m scripts.guiding_center.studies.magnetron.plot_discharge --run run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl --stem discharge --times 10,20,30,40
```

## Tests

`tests/test_magnetron.py`.

## Follow-ups

- `run_magnetron.py` still uses the in-memory `SnapshotSaver`; switching to the spilling saver of
  `runtime/observers.py` needs a validation run.
