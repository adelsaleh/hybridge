# Optional comparisons and field transfer

- `hdg_projection.py`: reconstruct exported cell polynomials and project them
  into HYBRIDGE DG spaces. Requires HYBRIDGE, NumPy, and SciPy; no DOLFINx runtime.
- `check_field_import.py`: real DOLFINx export → HYBRIDGE projection verification.
- `equilibrium.py`: compare two DOLFINx checkpoints on reference quadrature.

```bash
python -m projects.diocotron.comparisons.check_field_import
python -m projects.diocotron.comparisons.equilibrium --help
```

File validation and export belong to the DOLFINx implementation's
`checkpoint_data.py` and `checkpoint.py`. Projection belongs here. Neither
implementation imports this comparison layer to perform its own solves.
