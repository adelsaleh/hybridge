# Symmetric Triangle Quadrature Host Tests

This note summarizes the host-side tests added for symmetric triangle
quadrature experiments. The rule data now lives in
`hdgfem/core/quadrature.py` and is exercised by
`tests/test_symmetric_triangle_quadrature_host.py`. The default
`volume_quadrature="auto"` policy uses compact admissible Dunavant tables
through order 7. Above that table, it switches to the legacy collapsed Duffy
rule because the generated symmetric fallback has a larger point count. The
generated positive-weight symmetric rule remains available explicitly with
`volume_quadrature="symmetric"`; it is formed by expanding a minimally exact
Duffy rule into complete barycentric permutation orbits. The legacy collapsed
Duffy rule remains selectable with `volume_quadrature="duffy"`; passing
`volume_quad_1d` also selects that legacy path for backward compatibility.

## Main Result

For exact integration of polynomials of total degree `2p` on the HDGFEM
reference triangle, the tested Dunavant rules need:

| HDG order `p` | Exact degree `2p` | Symmetric rule points |
| ---: | ---: | ---: |
| 1 | 2 | 3 |
| 2 | 4 | 6 |
| 3 | 6 | 12 |
| 4 | 8 | 16 |
| 5 | 10 | 25 |
| 6 | 12 | 33 |
| 7 | 14 | 42 |

For `p=7`, this means a 42-point symmetric degree-14 rule is enough for exact
integration of degree-14 polynomials. The older collapsed Duffy default would
use `16 x 16 = 256` points, while the explicit generated symmetric fallback
uses 384 points.

## What Was Tested

The test module includes compact Dunavant rule data for exact degrees
`2, 4, 5, 6, 8, 10, 12, 14`. The rules are stored in unit-triangle barycentric
coordinates and then mapped to the HDGFEM reference triangle
`conv{(-1,-1), (1,-1), (-1,1)}`.

The tests check:

- monomial exactness through each rule's declared degree;
- integration of every basis mode in the full `P_{2p}` space for
  `bernstein`, `hier_C0`, and `dub_orth`, for `p=1..6`;
- mass and stiffness matrices for each `p=1..6` and each basis against a
  high-order collapsed Duffy reference;
- internal consistency of `ReferenceElementData` volume and face tensors from
  `hdgfem/core/quadrature.py`;
- finite basis and gradient values for the collapsed-coordinate bases at
  interior symmetric points and points very close to the top reference vertex.
- generated-rule symmetry and monomial exactness through `P_{2p}` when
  explicitly requested above the compact Dunavant table.

## Reference-Element Matrix Checks

The `ReferenceElementData` checks rebuild the same tensors by direct NumPy
contractions and compare them to cached fields:

- `MKrf`;
- `MKrf_inv`;
- `weighted_phi`;
- `weighted_phi_phi_flat`;
- `weighted_triple_phi_flat`;
- edge mass `M_rf_fc`;
- face coupling tables and their legacy aliases.

This verifies that the reference matrices are computed consistently from the
quadrature and basis tables. Production `ReferenceElementData` now uses the
automatic policy: compact degree-`2p` symmetric data where available, then
Duffy at higher order unless symmetric generation is explicitly requested.

## Exactness Caveat

Mathematically, all three supported element bases span polynomial spaces on
the triangle:

- Bernstein: direct barycentric polynomial basis;
- `hier_C0`: collapsed-coordinate hierarchical basis with removable
  singularities at the top vertex;
- `dub_orth`: Dubiner/Koornwinder-style collapsed-coordinate basis.

Therefore a triangle rule exact for `P_{2p}` integrates basis products
`phi_i phi_j` exactly for any of these bases in exact arithmetic. The
implementation-level risk is numerical evaluation near the collapsed-coordinate
vertex, not the mathematical polynomial degree. The current tests avoid putting
quadrature points exactly on that vertex and explicitly check finite values
near it.

For variable-coefficient or nonlinear weighted mass terms, `2p` exactness may
not be enough. For example, a coefficient represented in `P_p` multiplying
`phi_i phi_j` produces degree up to `3p`.

## Manufactured-Case Comparison

Matched runs can select either family through `DGSpace(...,
volume_quadrature="symmetric"|"duffy")` or the
`--volume-quadrature` option in both manufactured-case runners.

- Polynomial exactness check — diffusion-reaction `quadratic_poisson`, `p=4`,
  `mesh_size=0.35` (90 triangles): primal L2 errors were `4.3580e-14`
  (16-point symmetric) and `4.5993e-14` (100-point Duffy). The quadratic
  manufactured solution lies in the discrete polynomial space, so the
  roundoff-level errors verify polynomial reproduction.
- Oscillatory Poisson robustness check — `trigonometric_poisson_direct`,
  `p=6`, `mesh_size=0.06` on the disk (50,674 triangles): primal L2 errors
  were `2.2614e-09` (33-point symmetric) and `2.2855e-09` (196-point Duffy).
  The respective postprocessed primal L2 errors were `2.0029e-11` and
  `3.5666e-11`; Linf errors were `1.9716e-08` and `5.1869e-08`. This pair
  used Numba assembly, fused local assembly, boundary elimination, and the
  direct trace solver.
- Advection-reaction `test2`, `p=6`, `mesh_size=0.01`
  (92,552 triangles): L2 errors were `5.4341e-10` (33-point symmetric)
  and `5.6827e-10` (196-point Duffy).

The comparisons use identical meshes and solver settings within each pair.
The advection-reaction pair uses the fast production path: Numba assembly,
upwind-SCC trace ordering, ILU-preconditioned BICGSTAB, boundary elimination,
projected source/advection/reaction fields, ILU drop tolerance `1e-10`, and
ILU fill factor `35`. Both advection runs converged in one BICGSTAB iteration
with relative residuals below `2.7e-15`.

## Commands Run

Focused module:

```bash
.venv/bin/python -m pytest tests/test_symmetric_triangle_quadrature_host.py -q
```

Result:

```text
89 passed
```

Broader sanity run:

```bash
.venv/bin/python -m pytest \
  tests/test_symmetric_triangle_quadrature_host.py \
  tests/test_space.py \
  tests/test_diff_rea_solver_class.py \
  -q
```

Result:

```text
145 passed
```

## References

- `hdgfem/core/basis.py`: basis formulas and tabulation kernels.
- `hdgfem/core/quadrature.py`: compact and generated symmetric rules, legacy
  collapsed Duffy quadrature, and `ReferenceElementData` tensor construction.
- D. A. Dunavant, "High Degree Efficient Symmetrical Gaussian Quadrature Rules
  for the Triangle", International Journal for Numerical Methods in
  Engineering, 21, 1129-1148, 1985.
- J. N. Lyness and D. Jespersen, "Moderate Degree Symmetric Quadrature Rules
  for the Triangle", IMA Journal of Applied Mathematics, 15, 19-32, 1975.
- John Burkardt's Dunavant rule archive:
  https://people.sc.fsu.edu/~jburkardt/f_src/triangle_dunavant_rule/triangle_dunavant_rule.html
