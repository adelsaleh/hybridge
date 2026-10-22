# Optional comparisons and field transfer

- `hdg_projection.py`: reconstruct exported cell polynomials and project them
  into HYBRIDGE DG spaces. Requires HYBRIDGE, NumPy, and SciPy; no DOLFINx runtime.
- `check_field_import.py`: real DOLFINx export → HYBRIDGE projection verification.
- `equilibrium.py`: compare two DOLFINx checkpoints on reference quadrature.
- `discharge_supg.py`: the D7 collisional parallel-vane magnetron with CG Poisson and CG-SUPG density
  (same HYBRIDGE mesh and case, SI-BDF2 splitting, absorbing walls); records charge, consistent wall outflux
  and the charge-balance defect, the CG-SUPG side of the HDG comparison. FEniCSx interpreter.

```bash
python -m projects.diocotron.comparisons.check_field_import
python -m projects.diocotron.comparisons.equilibrium --help
```

File validation and export belong to the DOLFINx implementation's
`checkpoint_data.py` and `checkpoint.py`. Projection belongs here. Neither
implementation imports this comparison layer to perform its own solves.
