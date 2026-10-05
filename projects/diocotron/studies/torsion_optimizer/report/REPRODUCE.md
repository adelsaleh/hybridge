# Reproduction commands

Figure generation emits only high-resolution PNG publication assets. The
portable bundle and current generators contain, regenerate, and register PNG
figures only; SVG assets are not part of the current study output contract.

```bash
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py plan
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py run --campaign draft
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py aggregate
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py figures --stage preliminary
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py validate --require-terminal
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py report
python -m projects.diocotron.studies.torsion_optimizer.build_report
```

## Frozen-frontier threshold initialization

The dedicated frontier runner is
`projects/diocotron/dolfinx/torsion/initialization/frozen_frontier_run.py`.
It writes the complete binned hard-window Pareto frontier to
`RUN_DIR/logs/frozen_frontier.csv`, refines the selected pair with the actual
logistic window, and then uses the established source-homotopy projection.
Omitting `--frozen-leakage-cap-rel` selects the maximum frozen Jaccard point.
Supplying the option imposes a strict leakage cap relative to crisp target
area; infeasibility terminates the run and the cap is never enlarged.

Frontier counterparts of the high-DOF horseshoe and ITER commands are in
`projects/diocotron/studies/torsion_optimizer/configs/*_frozen_frontier_highdof_*_command.txt`.  The older
`*_fractional_phi_target_*` files remain available as historical controls.
