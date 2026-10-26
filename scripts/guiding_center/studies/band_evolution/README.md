# Band evolution study (E7)

Unperturbed control and band-mode seeded evolution of imported equilibrium bands (star, ugly
Pac-Man, ITER, horseshoe) from the DOLFINx equilibrium catalogue: preservation of the equilibrium by
the controls, growth of the seeded band mode, vortex census and the article figures of Section 7.2.
Side studies: the five-site seeds of the star band and the geometry-tied varicose lumps on every
domain. Every transfer is a discrete HDG equilibrium: `equilibrium_transfer_check.py` measures the
projected and the consistent transfer residuals in the Gram dual norm.

| Script | Role |
|---|---|
| `run_band_evolution.py` | E7 driver: `--kinds control,seed`, band modes per geometry (`--modes 'star=16;iter=18'`), site and lump seeds, snapshot schedule, spilling snapshots, profiles and movies; `--dry-run` prints the runs. Starts solves otherwise. |
| `analyze_band_evolution.py` | Preservation, band-mode growth and vortex census of control/seed pairs. No solve. |
| `plot_band_evolution_article.py` | Article figures and table (finished runs from the snapshot files, running ones from the spilled states). No solve. |
| `equilibrium_transfer_check.py` | Residuals of the projected and of the consistent transfer of one equilibrium. Poisson and Newton solves only (no time stepping). |

## Outputs

| Campaign | Directory | Status |
|---|---|---|
| E7 band evolution (four geometries, control and seed, T = 200) | `run_outputs/article/e7_band_evolution_20261019` | complete (`E7_DONE`; `runs_failed.jsonl` lists the failed attempts); `campaigns/gc_e7_band_evolution` |
| E7 five-site seeds of the star | `run_outputs/article/e7_five_sites_20261019` | complete (`X5_DONE`); `campaigns/gc_e7_five_sites` |
| E7 lumps on every domain | `run_outputs/article/e7_lumps_20261020` | complete (`LUMPS_DONE_a/b/c`); `campaigns/gc_e7_lumps` |

## Commands

Options only, then the E7 run form (`--dry-run` prints the runs; without it the run starts):

```bash
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.run_band_evolution --help
.venv/bin/python -u -m scripts.guiding_center.studies.band_evolution.run_band_evolution \
  --geometries star --kinds control,seed --modes 'star=16;uglypacman=16;iter=18;horseshoe=20' --edge-fraction 0.05 \
  --order 6 --dt 0.05 --final-time 200 --initial-projection-quad-1d 24 \
  --poisson-solver-atol 5e-14 --poisson-solver-rtol 5e-11 \
  --output-dir run_outputs/article/e7_band_evolution_20261019/runs --prefix e7 --skip-existing --verbosity 1 --dry-run
```

A smoke of the same configuration: `--steps 20 --snapshot-times 0.5,1 --output-dir run_outputs/article/e7_band_evolution_20261019/runs/smoke --prefix e7smoke --verbosity 2`.

Transfer check, analysis and figures (no time stepping):

```bash
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.equilibrium_transfer_check --help
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.analyze_band_evolution --runs run_outputs/article/e7_band_evolution_20261019/runs --output run_outputs/article/e7_band_evolution_20261019/analysis --prefix e7
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.plot_band_evolution_article --runs run_outputs/article/e7_band_evolution_20261019/runs --article ../gc-hdg-article --prefix e7
```

## Tests

`tests/test_band_evolution_driver.py`, `tests/test_band_evolution_article.py`, `tests/test_band_profiles.py`,
`tests/test_linearized_response.py`.
