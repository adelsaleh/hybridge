# GPU harmonic-Ritz polynomial preconditioner

## Why this stage follows the restart study

The T600 restart sweep showed that increasing the restart dimension from 30 to
75--100 cuts the ASM-GMRES time substantially.  One-pass CGS is the fastest
outer orthogonalization for the tested Poisson systems, while CGS2 preserves
basis orthogonality near machine precision.  The implementation therefore uses
these defaults:

- outer restarted GMRES: `orthogonalization="cgs"`;
- polynomial spectral setup: `setup_orthogonalization="cgs2"`.

Harmonic Ritz roots are more sensitive to a defective Arnoldi basis than the
final restarted solve, so the safer two-pass method is retained for setup.

## Mathematical action

For no base preconditioner, the inverse polynomial is

\[
p_{P-1}(A)=\sum_{j=1}^{P}
\left[\prod_{k=1}^{j-1}\left(I-\frac{A}{\theta_k}\right)\right]
\frac{1}{\theta_j}.
\]

With block-Jacobi or ASM, the hybrid action is

\[
P_{\mathrm{hybrid}}^{-1}y
=p_{P-1}(M^{-1}A)M^{-1}y.
\]

Every polynomial degree therefore costs one global face-dense matvec and, for
BJ-PP or ASM-PP, one base-preconditioner application.  There is one additional
base application for the initial `M^{-1}y`.

## Spectral setup

`setup_polynomial_preconditioner_cupy` runs one matrix-free Arnoldi cycle on
`A` or `M^{-1}A`.  Large vectors stay on the GPU.  Only the short projection
coefficient vectors and the final Hessenberg matrix are used on the CPU.

For `Hbar` of shape `(P+1,P)`, harmonic Ritz values are the eigenvalues of

\[
H_P+h_{P+1,P}^2 H_P^{-T}e_Pe_P^T.
\]

The roots are reordered by a deterministic greedy Leja heuristic.  Complex
conjugates are kept adjacent.

## Real GPU recurrence

For a real root `theta`, one fused kernel evaluates

```text
z += q/theta
q -= Bq/theta
```

For a pair `a+ib`, `a-ib`, two applications of `B` are followed by one fused
kernel:

\[
z\leftarrow z+\frac{2a q-Bq}{a^2+b^2},
\]

\[
q\leftarrow q-
\frac{2a}{a^2+b^2}Bq+
\frac{1}{a^2+b^2}B^2q.
\]

The linear `Bq` term in the second update is necessary: it follows directly
from expanding

\[
(I-B/\bar\theta)(I-B/\theta)q.
\]

The paper's printed paired update omits that term.  The implementation uses the
algebraically complete recurrence and tests it against explicit complex
arithmetic.

## Device storage

Application reuses four vectors of length `num_dofs`:

- `q`;
- `Bq`;
- `B2q`;
- temporary output of `Aq` before an optional base preconditioner.

The caller supplies the final output vector.  No allocation occurs inside
`apply_into`.

## Basic use

```python
from hdgfem.backends.cupy_polynomial import CuPyPolynomialPreconditioner

asm_poly = CuPyPolynomialPreconditioner.from_operator(
    operator,
    degree=8,
    base_preconditioner=asm,
    seed=1729,
    setup_orthogonalization="cgs2",
)

result = restarted_gmres_cupy(
    operator,
    rhs_gpu,
    restart=100,
    max_iterations=2000,
    rtol=1.0e-8,
    preconditioner=asm_poly,
    orthogonalization="cgs",
)
```

## T600 degree study

For the previously studied `64x64, p=1` case:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_polynomial.py \
    --mesh 64 \
    --order 1 \
    --degrees 2 4 6 8 10 12 \
    --methods asm asm_poly \
    --restart 75 \
    --max-iterations 2000 \
    --outer-orthogonalization cgs \
    --setup-orthogonalization cgs2 \
    --output-prefix results/t600_asm_poly_64_p1
```

For `64x64, p=4`, where restart 100 was best:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_polynomial.py \
    --mesh 64 \
    --order 4 \
    --degrees 2 4 6 8 10 12 \
    --methods asm asm_poly \
    --restart 100 \
    --max-iterations 2000 \
    --outer-orthogonalization cgs \
    --setup-orthogonalization cgs2 \
    --output-prefix results/t600_asm_poly_64_p4
```

Select the polynomial degree by total solve time, not iteration count alone.
The report separates setup, one preconditioner application, and the outer
solve.
