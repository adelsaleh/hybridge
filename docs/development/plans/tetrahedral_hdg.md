# HDG On Tetrahedra: Reference Simplices, Bases And Reference Integrals

Date: 2026-11-22. Requested by the user. Status: milestone 1 in progress; nothing in
three dimensions is supported yet. Tracked in `TODO.md`, "Tetrahedral Elements".

## Goal

HYBRIDGE solves HDG problems on affine triangles only. This plan generalizes it to affine
tetrahedra. The first milestone builds the reference triangle and the reference tetrahedron,
several polynomial bases on them, and the reference integrals from which every HDG local block
is formed, for any polynomial degree `p`. It then tests these integrals on HDG problems of
increasing difficulty, one stage at a time.

Notation: `d` is the dimension (2 or 3). The volume space `P_p(K)` has
`n_u = C(p+d, d)` functions and the trace space on one face has `n_t = C(p+d-1, d-1)`
functions. On the tetrahedron, `(n_u, n_t)` is `(4, 3)` at `p = 1`, `(20, 10)` at `p = 3`,
`(56, 21)` at `p = 5` and `(286, 66)` at `p = 10`.

## Reference simplices

- The reference triangle is `conv{(-1,-1), (1,-1), (-1,1)}` (area 2), as today. The reference
  tetrahedron is `conv{(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)}` (volume 4/3).
- Triangle faces keep the present numbering: face `f` joins vertex `f` to vertex `(f+1) mod 3`,
  parametrized by `t` in `[-1, 1]` from the first to the second vertex. Tetrahedron face `f` is
  the face opposite vertex `f`, with its vertices in increasing order.
- Face integrals are tabulated on the reference face `F^`. The ratio `|F_f| / |F^|` is stored
  separately: `(1, sqrt 2, 1)` on the triangle and `(sqrt 3, 1, 1, 1)` on the tetrahedron.
- A face shared by two elements is seen by each element through a permutation `sigma` of the
  face vertices: two permutations for a triangle edge, six for a tetrahedron face. The trace
  basis is defined in the global vertex order of the face; element `K` sees it through
  `sigma_{K,f}`, which maps the local vertex order to the global one.

## Quadrature

- Collapsed Gauss-Jacobi rules exist for `d = 1, 2, 3` and any exact degree `k`. The Duffy
  Jacobian is absorbed into Gauss-Jacobi weights `(1-x)^1` and `(1-x)^2`, so `ceil((k+1)/2)`
  points per direction suffice. All weights are positive.
- Tetrahedron faces use the existing symmetric triangle rules. They are invariant under the six
  vertex permutations, so a face orientation permutes the quadrature points.
- By default, the volume and face rules on the tetrahedron are exact for degree `2p`.

## Bases

Modal bases follow Karniadakis and Sherwin, *Spectral/hp Element Methods for CFD*, 2nd ed.,
chapter 3:

| Name | Triangle | Tetrahedron |
|---|---|---|
| `dub_orth` | existing orthogonal basis (not normalized) | collapsed-coordinate orthogonal basis, same normalization |
| `orthonormal` | `dub_orth` scaled so that the mass matrix is the identity | same |
| `bernstein` | existing | `C(p; alpha) lambda^alpha`, `|alpha| = p` |
| `hier_C0` | existing modified C0 basis | modified basis: vertex, edge, face and interior modes |

The collapsed coordinates are singular at the top vertex and on one edge of the tetrahedron,
and nodal sets and face quadrature place points there. The tetrahedron bases are therefore
evaluated in their equivalent barycentric form, with scaled Jacobi recurrences
`t^n P_n^{(a,b)}(x/t)` that contain no division (Kirby, ACM TOMS 37, 2010).

Nodal Lagrange bases `lagrange_<family>` are built on the orthonormal Vandermonde matrix:

- **`equispaced`**: the equispaced lattice.
- **`warp_blend`**: warp and blend nodes (Warburton 2006; Hesthaven and Warburton 2008).
- **`recursive`**: recursive nodes (Isaac, SIAM J. Sci. Comput. 42, 2020).
- **`fekete`**: interior nodes that maximize `|det V|`. Edges carry Gauss-Lobatto points and
  faces carry the triangle `fekete` set, so the restriction of the volume nodes to a face is
  the face node set.
- **`fekete_free`**: all nodes maximize `|det V|`, the boundary ones included.

The Fekete sets are computed once by an optimization over symmetric orbits and stored as package
data: triangle `p <= 15`, tetrahedron `p <= 10`. Above these degrees they are computed on request.

The trace basis on a face is one of these bases on the simplex of one dimension lower. On the
triangle, `lagrange_*` in one dimension is the Gauss-Lobatto Lagrange basis (`legacy-lagrange`),
and `dub_orth` in one dimension is the Legendre basis (`legendre-modal`).

## Reference integrals

| Block | Table | Shape | Entry |
|---|---|---|---|
| uu | mass `M` | `(n_u, n_u)` | `int_K^ phi_i phi_j` |
| uu | face mass `E^f` | `(d+1, n_u, n_u)` | `int_F^ (phi_i phi_j) o F_f` |
| u-uhat | element-trace coupling | `(d!, d+1, n_u, n_t)` | `int_F^ (phi_i o F_f) (mu_a o sigma)` |
| uhat-u | trace-element coupling | `(d! (d+1), n_t, n_u)` | the same with the trace function as test function |
| uhat-uhat | trace mass `G` | `(n_t, n_t)` | `int_F^ mu_a mu_b` |
| uu | derivative `D^k` | `(d, n_u, n_u)` | `int_K^ phi_i d_k phi_j` |
| uu | stiffness `K^{kl}` | `(d, d, n_u, n_u)` | `int_K^ d_k phi_i d_l phi_j` |

`ReferenceElementData` gains a dimension and the constructor `ReferenceElementData.tetrahedron`.
Its two-dimensional tables are unchanged. The trace-dependent tables are built by one function
for one volume basis and one trace basis.

## Stages

Each stage is finished and reported before the next starts. Blocks follow the sign convention of
`TODO.md`, "Form-Driven HDG Assembly": `A u - B uhat = f`, `D uhat - C u = g`, condensed
`S = D - C A^{-1} B`.

1. **Stage A: `alpha u = f` on the reference triangle and tetrahedron.** The local problem is
   `(alpha u, v)_K + <tau (u - uhat), v>_dK = (f, v)_K`. It gives
   `A = alpha M + tau sum_f r_f E^f`, `B` and `C` from the couplings scaled by `tau r_f`, and
   `D = tau r_f G`. Checks:
   - a polynomial `u*` of degree `p` is reproduced exactly;
   - `A` and `S` are symmetric positive definite;
   - `S` does not depend on the volume basis;
   - at `tau = 0` the solution is the L2 projection;
   - a face orientation changes `S` by the orientation matrices;
   - on the triangle, the blocks equal those of the present 2D assembly with zero velocity.
2. **Stage B: the geometry of `alpha u = f`.** Affine elements, then patches of two and of six
   tetrahedra (two and four triangles) with every face orientation. Checks: polynomial
   reproduction, positive definiteness of the global condensed matrix, and invariance under
   relabeling of element vertices.
3. **Stage C: advection `beta . grad u + alpha u = f`.** Derivative tables and their
   integration-by-parts identity `D^k + (D^k)^T = sum_f n_{f,k} r_f E^f`, then the upwind HDG
   blocks of the present transport solver, on the reference element and then with geometry.
4. **Stage D: diffusion in mixed form.** Stiffness tables, then the mixed HDG blocks of the
   present diffusion solver, on the reference element and then with geometry. For `alpha = 0`,
   the condensed matrix annihilates constant traces.

## Later milestones

1. Tetrahedral `DGMesh`: box meshes cut into six tetrahedra per cube and Gmsh import; face
   orientation relative to sorted global vertex numbers.
2. Spaces, fields and trace spaces in three dimensions.
3. NumPy assembly, static condensation and global solve. Accepted when manufactured solutions
   converge at rate `p+1` for `u` and `q` and at rate `p+2` after postprocessing.
4. Numba, then CuPy and raw CUDA. AMGX face blocks have size `n_t`: the multicolor DILU smoother
   accepts block sizes 1 to 10, which covers `p <= 3`.

## Acceptance

Milestone 1 is accepted when stages A to D pass on the triangle and the tetrahedron for every
basis, and when hashes of every existing two-dimensional table show that no 2D result changed.
