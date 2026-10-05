# DOLFINx equilibrium workflows

This directory contains the optional DOLFINx research solvers and their
checkpoint adapters. They are checkout scripts, not modules of the installed
`hdgfem` library. Activate a matching FEniCSx/PETSc/MPI environment and run
commands from the repository root.

## Fixed-threshold equilibrium bands

The [equiband package](equiband) solves for the equilibrium potential and
threshold midpoint at a fixed threshold-space width. Its center-distance
coordinate is normalized torsion-flow arclength `zeta_T=s/L`; `rho` denotes
guiding-center density only.
The default smoothing is width-relative: `relative_epsilon=0.08` resolves
`epsilon=0.08*threshold_width_delta`. The startup log reports the mode,
effective epsilon and window peak. Old absolute-mode runs require their
original configuration for restart; updated examples require fresh runs.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic --write-vtk \
  --plot --plot-mode nonblocking -v 2 --save-terminal-log
```

The CLI defaults to linked live PyVista panels and maximum verbosity 2.
Press Enter in the terminal or plot after the final state to finish.
Use `--plot-mode blocking` for per-state pauses, `--no-plot` for batch runs,
or `--plot-off-screen --save-frames` for headless PNG output.
Live rendering reuses contour actors and fixed-size annotations; the
`--plot-min-interval` option controls its wall-clock cadence.

Target runs default to pseudo-arclength continuation, oriented initially
toward the requested distance. The midpoint can turn through a simple fold
without relaxing the two-interface-band guards. No `--m-stop` is needed;
`--arc-max-steps` / `--arc-max-length` bound the search, which stops at its
first target. Explicit midpoint scans use `--continuation midpoint --m-stop`.

`--save-terminal-log` records Python/native stdout and stderr under
`OUTPUT/logs/SESSION/terminal_rankNNNN.log`, plus per-rank exit status JSON.
Restart sessions preserve previous transcripts. Elapsed times, physical
inputs, scan direction, guard rejections and final outcomes are explicit.
Newton iteration monitoring is restored before each solve, including after
failed trials and for the bordered arclength corrector.

Existing output directories trigger a warning and a fresh/resume/cancel
prompt. `--overwrite-output` skips the prompt, archives the entire old run
to a timestamped sibling backup, and starts fresh at the requested path.
`--restart` skips the prompt and resumes compatible checkpoints. Do not
reuse a directory that another process is writing.
See the [user guide](../docs/equiband.md) for the
functionals, branch guards, MPI commands and limitations, and the
[examples](../examples/equiband/README.md) for configurations and an
environment specification. Equiband itself does not import `hdgfem`.

## Curved horseshoe torsion-center audit

The independent [torsion_center_audit.py](geometry/center_audit.py) checks
torsion maxima and gradient zeros without invoking the equilibrium solver or
ray atlas. It supports solution degrees P2 through P6 on affine or curved
triangles. The installed DOLFINx Gmsh importer supports quadratic/cubic
triangular coordinate maps; **geometry degree and solution degree are separate**.

Generate cubic-curved triangles on the original horseshoe CAD boundary:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.geometry.canonical horseshoe \
  --mesh-size 0.03 --geometry-degree 3 \
  --output projects/diocotron/runs/equiband/meshes/horseshoe_h003_g3.msh
```

Then compare torsion and recovery orders:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.geometry.center_audit \
  --mesh projects/diocotron/runs/equiband/meshes/horseshoe_h003_g3.msh \
  --degrees 2 3 4 6 --output projects/diocotron/runs/equiband/horseshoe_center_hp \
  --save-terminal-log -v 2
```

By default each degree-p torsion solve is compared with CG1 and CG(p-1)
L2 gradient recovery; `--recovery-degrees` explicitly selects alternatives.
The audit uses batched NumPy polynomial operations and MPI-owned cells.
It writes `audit.json`, `degree_P.json` and full native/Python per-rank
terminal logs. Verbosity 2 and terminal capture are defaults. Existing output
can be archived at the prompt or with `--overwrite-output`; this diagnostic
has no equilibrium checkpoint restart. It does not produce an equilibrium plot.

For a finer mesh, regenerate under a new filename with a smaller `--mesh-size`
and pass that file to the audit. The h=0.0075 P2/P4 check used four physical
cores with one numerical-library thread each:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 4 \
python -m projects.diocotron.dolfinx.geometry.center_audit \
  --mesh projects/diocotron/runs/equiband/meshes/horseshoe_h00075_g3.msh \
  --degrees 2 4 --output projects/diocotron/runs/equiband/horseshoe_center_hp_fine \
  --save-terminal-log -v 2
```

The production equiband ray evaluator now accepts affine, quadratic and cubic
triangular coordinate maps, torsion degrees P2--P6 and continuous recovered
gradients CG1--CG5. Physical positions, tangent speeds and arclengths use the
full curved map. Its cyclic-order audit synchronizes the boundary-seeded rays
on common `T/T_max` slices; the strongly contracted central region is recorded
as an unresolved flow core and the innermost band interface must stay outside
it. The independent audit remains useful because it screens all cells for raw
and recovered critical points without depending on a target equilibrium.

The manual generator commands above remain appropriate for the standalone
center audit, whose interface consumes a named `.msh`. Equiband itself now
uses a deterministic MPI-safe cache for canonical geometries: set
`geometry="horseshoe"`, `mesh_size` and `geometry_degree` in its configuration
and do not pass `mesh_file`. Its first run generates under
`.cache/hdgfem/dolfinx_meshes`; later runs authenticate and load the same
artifact. Changing size or geometry order selects a new key. Explicit
`geometry="msh"`/`--mesh-file` mode remains available for external meshes and
never remeshes a file from a changed config number.

For the validated horseshoe command, 256/512/1024-ray comparison, P6/CG5
check, smaller-mesh atlas check and direct width/distance overrides, see the
[equiband examples](../examples/equiband/README.md#curved-horseshoe-target-and-refinement-checks)
and the
[horseshoe h/p and target results](../studies/equiband_validation/equiband_horseshoe_validation_20260909.md).
Candidate counts and unresolved cells are reported; multistart root searches
and common-level flow checks are numerical evidence, not mathematical
uniqueness proofs.

## Portable checkpoint adapters

- [checkpoint.py](checkpoint.py) writes generic scalar-field
  and equilibrium v2 checkpoints from DOLFINx functions.
- [checkpoint_data.py](checkpoint_data.py) validates and reads portable
  checkpoint arrays without importing either solver library.
- [The HDG projection adapter](../comparisons/hdg_projection.py) reconstructs
  the fields in native HDGFEM spaces. Only this optional comparison adapter
  depends on HDGFEM projection utilities.

Import these helpers explicitly from the script layer:

```python
from projects.diocotron.dolfinx.checkpoint import write_equilibrium_checkpoint_v2
from projects.diocotron.comparisons.hdg_projection import load_dolfinx_equilibrium
```

The old `hdgfem.io` exports are intentionally removed. Existing v2 checkpoint
format identifiers and data layouts are unchanged; no file conversion is
needed. To exercise real DOLFINx export and HDGFEM import on a small mesh:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python projects/diocotron/comparisons/check_field_import.py
```
