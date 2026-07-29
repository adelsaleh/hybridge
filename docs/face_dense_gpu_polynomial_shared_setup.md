# Shared Arnoldi setup and polynomial-degree amortization

## Motivation from the T600 results

The first degree sweep showed that ASM-polynomial preconditioning is already
substantially faster than ASM alone.

For the `64 x 64`, `p=1` case:

- ASM: 375 iterations, 481.842 ms;
- fastest solve-only polynomial: degree 18, 19 iterations, 82.645 ms;
- fastest cold `setup + solve`: degree 14, 105.393 ms.

For the `64 x 64`, `p=4` case:

- ASM: 400 iterations, 628.333 ms;
- fastest solve-only polynomial: degree 18, 22 iterations, 206.680 ms;
- fastest cold `setup + solve`: degree 12, 243.933 ms.

The preferred degree therefore depends on whether setup is paid once or
amortized over several right-hand sides.

## One maximum-degree Arnoldi probe

An Arnoldi factorization of dimension `Pmax` contains every leading relation
needed for lower dimensions:

\[
\bar H_d = \bar H_{P_{\max}}[0:d+1,0:d],
\qquad d \le P_{\max}.
\]

For a fixed initial vector and orthogonalization mode, the first `d` Arnoldi
steps are the same as an independent dimension-`d` run.  Therefore a degree
sweep does not need to repeat matrix-vector products and ASM applications for
every candidate.

The new API is:

```python
probe = setup_polynomial_arnoldi_probe_cupy(
    operator,
    maximum_degree=20,
    base_preconditioner=asm,
    seed=1729,
    orthogonalization="cgs2",
)

poly_12 = CuPyPolynomialPreconditioner.from_probe(
    operator,
    probe=probe,
    degree=12,
    base_preconditioner=asm,
)

poly_18 = CuPyPolynomialPreconditioner.from_probe(
    operator,
    probe=probe,
    degree=18,
    base_preconditioner=asm,
)
```

The large GPU Krylov basis is released after the probe.  Only the small host
Hessenberg and Gram matrices are retained.  Each degree candidate computes
its own harmonic Ritz values and conjugate-preserving Leja order from the
appropriate leading submatrix.

The probe is tied to the exact operator and base-preconditioner objects.  The
constructor rejects attempts to reuse it with another matrix or another ASM
object.

## Separating CUDA JIT from numerical setup

The first polynomial candidate previously included one-time `RawModule`
compilation in its setup time.  This explains the approximately 64--66 ms
setup time reported for degree 2, while later candidates were much cheaper.

Polynomial update kernels are now cached per CUDA device and dtype.  The
benchmark calls:

```python
initialize_polynomial_kernels_cupy(
    dtype=operator.dtype,
    device_id=operator.device_id,
)
```

before numerical timing and reports this one-time initialization separately.
Normal solver use remains lazy and requires no explicit warm-up call.

## Amortized selection criterion

For `N_rhs` systems using the same matrix and polynomial object, the relevant
average cost is

\[
T_{\mathrm{avg}}(P,N_{\mathrm{rhs}})
=
T_{\mathrm{solve}}(P)
+
\frac{T_{\mathrm{setup}}(P)}{N_{\mathrm{rhs}}}.
\]

Using the first independent-setup measurements:

- `p=1`: degree 14 is best for one to roughly five right-hand sides; degree 18
  becomes preferable for about ten or more;
- `p=4`: degree 12 is best for one right-hand side, while degree 18 becomes
  preferable once setup is amortized over at least two.

These are T600/Poisson-specific conclusions.  The target GPU and nonlinear
operators must be measured independently.

## Updated benchmark

The validation script defaults to shared setup:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_polynomial.py \
    --mesh 64 \
    --order 4 \
    --degrees 2 4 6 8 10 12 14 16 18 20 \
    --methods asm asm_poly \
    --setup-mode shared \
    --rhs-counts 1 2 5 10 20 \
    --restart 100 \
    --max-iterations 2000 \
    --outer-orthogonalization cgs \
    --setup-orthogonalization cgs2 \
    --output-prefix results/t600_asm_poly_shared_64_p4
```

It reports:

- one-time CUDA kernel initialization;
- shared Arnoldi probe time;
- per-candidate root extraction and workspace construction;
- polynomial application time;
- GMRES time and convergence;
- cold `setup + solve` time;
- best degree for each requested number of right-hand sides.

Use `--setup-mode independent` to reproduce the previous behavior and verify
that shared-probe and independent roots lead to the same solution.

## Important interpretation

Shared setup accelerates a *degree study* and makes online candidate
construction cheap.  It does not remove the cost of solving with every
candidate.  After an offline degree has been selected for a stable problem
family, production runs should normally construct only that degree and reuse
the resulting preconditioner for every compatible right-hand side.
