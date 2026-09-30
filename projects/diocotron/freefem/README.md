# FreeFEM implementations

- `equilibrium/`: standalone equilibrium programs.
- `equilibrium/strategy_a/`: the existing Strategy A variants and analysis.
- `guiding_center/`: guiding-center evolution programs.
- `guiding_center/variants/`: the former `ff/GC/` programs. Distinct versions
  have been retained even when their filenames also occur in the parent.
- `msh/`: existing geometry inputs, including the canonical ITER wall.

These programs are independent of the Python solver implementations.
Their existing relative data paths assume the FreeFEM project directory as
the working directory. For example:

```bash
cd projects/diocotron/freefem
FreeFem++ equilibrium/strategyA_torsion_initialized_newton.edp
```

The `out`, `logs`, `runs`, and `vtk` directory links point into
`../runs/freefem/`, retaining existing program conventions while keeping raw
results outside the source directories. These ignored directories must exist
before running programs that do not create their own output directories:

```bash
mkdir -p ../runs/freefem/{out,logs,runs,vtk}
```

The mathematical note is maintained in [the project docs](../docs/freefem_equilibria.tex).
