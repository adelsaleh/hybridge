# Diocotron equilibria and guiding-center dynamics

This checkout project owns the diocotron application: its DOLFINx, native HDG,
and FreeFEM implementations, scientific studies, and reports. The general
`hybridge` library remains at the repository root and is packaged independently.
Run the commands below from the repository root.

Generated figures and frames (PNG) of the studies are not distributed with the
repository; the reports under `studies/*/report/` reference a `figures/`
directory that is recreated by rerunning the corresponding study. Study run
outputs live under the local, untracked `run_outputs/`.

## Implementations

| Directory | Responsibility | Numerical environment |
| --- | --- | --- |
| [dolfinx/](dolfinx/README.md) | Equiband, torsion-based equilibrium optimization, CG/SUPG evolution | FEniCSx/PETSc/MPI; no HYBRIDGE imports |
| [hdg/](hdg/README.md) | Native HDG equilibrium experiments | HYBRIDGE and its selected optional backends |
| [freefem/](freefem/README.md) | FreeFEM equilibrium and evolution programs | FreeFEM and the program's required plugins |
| [comparisons/](comparisons/README.md) | Checkpoint projection and implementation comparisons | Dependencies of the particular comparison |

The implementations have separate numerical models and configuration policies.
Neither backend imports the other's solvers. The comparison code may use
HYBRIDGE and lightweight DOLFINx checkpoint data; parsing those files does not
import the DOLFINx runtime. Existing checkpoint v2 identifiers are preserved.

## Start a run

Use [environments/dolfinx.yml](environments/dolfinx.yml) for the FEniCSx stack
and [the equiband examples](examples/equiband/README.md) for full commands.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband \
  --config projects/diocotron/examples/equiband/disk_logistic.toml \
  --output projects/diocotron/runs/equiband/disk_logistic --no-plot

python -m projects.diocotron.dolfinx.torsion.optimization.homotopy --help
python -m projects.diocotron.hdg.equilibrium.newton --help
```

Runs with `--linear-solver mumps` use one numerical-library thread per MPI
rank; the best rank count depends on the mesh size, the order and the machine,
so benchmark it before long runs. HDG GPU configuration continues to use
the library's [backend configurations](../../configs/amgx/README.md).

## Studies and output ownership

- [docs/](docs/README.md): mathematical derivations and solver usage.
- [studies/torsion_optimizer/](studies/torsion_optimizer/REPRODUCE.md): case
  planning, analysis, launch configurations, and the numerical-tests article.
- [studies/equiband_validation/](studies/equiband_validation/README.md):
  disk, star, and horseshoe validation evidence.
- [studies/stationarity/](studies/stationarity/README.md): equilibrium
  stationarity and guiding-center evolution reports.
- [studies/strategy_a/](studies/strategy_a/README.md): archived parameter study.
- `runs/<study>/<run-id>/`: raw results, logs, checkpoints, and frames; ignored
  by Git. Publication evidence selected for a study belongs with its report.
- `build/<study>/`: generated PDFs, LaTeX auxiliaries, and archived build
  products; ignored by Git.

Compile the existing article without rerunning simulations or regenerating
its manually reviewed sections:

```bash
python -m projects.diocotron.studies.torsion_optimizer.build_report
```

The result is `projects/diocotron/build/torsion_optimizer/numerical_tests.pdf`.
The [migration notes](docs/layout_migration.md) describe old locations,
preserved artifacts, and compatibility with archived paths.

## Verification

The root test suite owns HYBRIDGE. The application suite is selected explicitly
and uses the FEniCSx environment for its finite-element and MPI checks:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest projects/diocotron/tests
```

On a headless machine, set `VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow`
when exercising the real off-screen rendering tests. MPI subprocess tests are
explicit opt-ins: `EQUIBAND_RUN_MPI_TESTS=1` and
`HYBRIDGE_RUN_DOLFINX_INTEGRATION=1`. Use the MPI launcher belonging to the active
FEniCSx environment.

Pure Python/NumPy tests and checkpoint projection tests can also be selected
individually in an HYBRIDGE environment. Optional comparisons involving both
libraries are isolated under `tests/comparisons/`. No application package is
included in the HYBRIDGE wheel.
