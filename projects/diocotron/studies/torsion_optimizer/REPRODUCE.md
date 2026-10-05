# Torsion optimizer study

Run all commands from the repository root with the FEniCSx environment active.
The numerical algorithms live in `dolfinx/torsion`; this study owns campaign
definitions, scheduling, analysis, figure generation, and its report.

```bash
python -m projects.diocotron.studies.torsion_optimizer.run --help
python -m projects.diocotron.studies.torsion_optimizer.run run --dry-run --limit 1
python -m projects.diocotron.studies.torsion_optimizer.build_report
```

The last command only compiles the existing report. Its default output is
`projects/diocotron/build/torsion_optimizer/numerical_tests.pdf`.

[report/REPRODUCE.md](report/REPRODUCE.md) contains the original campaign steps.
`plan`, `run`, `aggregate`, `figures`, and `report` modify study results or
generated report sources and should be invoked intentionally. Use `--manifest`,
`--bundle`, and `--run-root` before the subcommand to select an independent
campaign. Existing completed-run evidence is retained in the moved manifest.

Preserved launch commands are in [configs/](configs/). Raw results default to
`projects/diocotron/runs/torsion_optimizer/`. Historical paths inside saved
manifests are resolved through the project's relocation index at read time.
