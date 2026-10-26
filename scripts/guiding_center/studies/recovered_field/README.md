# Recovered-field diagnostics

Bounded matrix diagnostics of the electric-field recovery order behind BDF2 with reduced-order
Poisson (`docs/recovered_field.md`): manufactured data on a fixed unit square (at most 512
triangles), NumPy assembly and SciPy direct solves, no time scheme. `NUMBA_DISABLE_JIT=1` is
mandatory for the two order checks: they must not compile kernels.

| Script | Role |
|---|---|
| `check_field_recovery_order.py` | Recovery order against the degree-p reference solve on nested subdivisions (`--subdivisions`, `--jitter`, `--stabilization`). |
| `check_oversampled_recovery_order.py` | Experimental oversampled recovery on a perturbed square mesh (one degree-(p-1) solve and four degree-p patch solves). |
| `compare_field_recovery.py` | One-element recovery diagnostic with a prescribed exact trace. |

## Commands

```bash
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.check_field_recovery_order --help
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.check_field_recovery_order --density-order 3 --subdivisions 2 4 8
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.check_oversampled_recovery_order --help
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.compare_field_recovery --density-order 3
```

## Outputs

Printed tables only; nothing is saved. The recovered-field evidence of the article is in
`docs/recovered_field.md` ("Electric-field convergence diagnostic").
