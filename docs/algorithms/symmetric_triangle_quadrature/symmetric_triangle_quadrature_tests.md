# Symmetric Triangle Quadrature Host Tests

This note summarizes the host-side tests added for symmetric triangle
quadrature experiments. The implementation lives in
`tests/test_symmetric_triangle_quadrature_host.py` and is currently test-only:
no production quadrature or GPU assembly path has been changed to use these
rules yet.

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

For `p=6`, this means a 33-point symmetric degree-12 rule is enough for exact
integration of degree-12 polynomials. The number 91 is the dimension of
`P_12`, not the number of quadrature points required by this symmetric
Gaussian rule. The older collapsed Duffy setting `volume_quad_1d=12` uses
`12 x 12 = 144` points.

## What Was Tested

The test module includes compact Dunavant rule data for exact degrees
`2, 4, 5, 6, 8, 10, 12`. The rules are stored in unit-triangle barycentric
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

This verifies that the current reference matrices are computed consistently
from the quadrature and basis tables. It does not prove that every tensor is
integrated with a minimal exact rule, because `ReferenceElementData` still uses
the existing collapsed tensor-product Duffy rule.

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

## Commands Run

Focused module:

```bash
.venv/bin/python -m pytest tests/test_symmetric_triangle_quadrature_host.py -q
```

Result:

```text
71 passed in 9.21s
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
124 passed in 11.22s
```

## References

- `hdgfem/core/basis.py`: basis formulas and tabulation kernels.
- `hdgfem/core/quadrature.py`: current collapsed Duffy reference quadrature
  and `ReferenceElementData` tensor construction.
- D. A. Dunavant, "High Degree Efficient Symmetrical Gaussian Quadrature Rules
  for the Triangle", International Journal for Numerical Methods in
  Engineering, 21, 1129-1148, 1985.
- J. N. Lyness and D. Jespersen, "Moderate Degree Symmetric Quadrature Rules
  for the Triangle", IMA Journal of Applied Mathematics, 15, 19-32, 1975.
- John Burkardt's Dunavant rule archive:
  https://people.sc.fsu.edu/~jburkardt/f_src/triangle_dunavant_rule/triangle_dunavant_rule.html
