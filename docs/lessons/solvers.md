# Lessons from the linear solver studies

These lessons come from solving the trace systems of
advection-diffusion-reaction (ADR) and Poisson problems with iterative
solvers on a GPU and with sparse direct LU on CPU nodes. The detailed solver
kernel measurements are in
[`campaigns/solver_h200`](../../campaigns/solver_h200/README.md); the kernel
settings they led to are in
[kernel tuning by GPU architecture](kernel_tuning_by_architecture.md).

A solve is *repeated* when it reuses a preconditioner that is already set
up, and *warm* when it also starts from the solution of a nearby problem.
DOF means degree of freedom of the trace system.

## Iterative against direct

1. **Up to about 20 million DOFs, sparse direct LU on a CPU node was not
   beaten by the iterative GPU solvers.** PARDISO on 96 cores factored and
   solved 1.0 million DOFs in 2.2 s and 20 million in 36 s, and a solve that
   reuses the factorization took 0.23 s and 5 s. The best preconditioned
   GMRES on the GPU needed 120 s and 426 iterations for a cold solve at 15
   million DOFs. Direct LU also solved the stress cases on which every
   iterative candidate stalled.
2. **Direct LU has its own limits at this size.** At about 20 million DOFs
   the residual stopped at 1.1 to 1.4e-10, just above the target; the number
   of nonzeros of the factor no longer fits in a 32-bit count; and the memory
   in use was 1.4 to 1.6 times the estimate.
3. **A solve that does not reach its tolerance is a result, not a crash.**
   Record it with its status and its residual and go on. Use one rule for
   the status everywhere; two rules produced "not converged" in one table
   and "numerical failure" in another for the same run.

## Stopping tests

4. **Meeting the residual target does not imply meeting the error bound.**
   With a DILU preconditioner, variable-velocity cases reached a relative
   residual of 1e-10 and missed an L2 error bound of 2e-8. A stricter inner
   tolerance of 1e-12 met the bound in all six cases for 6 to 25% more
   iterations.
5. **A warm solve must stop at the requested residual.** A stopping test
   relative to the residual at the start of each restart cycle makes a
   solve that starts close to the solution iterate to round-off. Scaling the
   test to the requested target (`estimate_target="true_target"`) took one
   Schwarz-preconditioned solve from 58 to 33 iterations. It leaves less
   margin: some high-degree runs then met the residual and missed the L2
   bound.
6. **Stop relative to the right-hand side when warm starts matter.** A test
   relative to the initial residual gains nothing from a good initial guess;
   it only makes the answer tighter.

## Kernels inside the solver

7. **Solver kernels read much and compute little.** The face-dense operator
   and the Schwarz application do about 0.25 floating-point operations per
   byte. The order in which the matrices are stored decided their speed
   (1.8 and 2.2 times faster once stored in reading order), and the number
   of threads per block did not.
8. **Choose the variant per kernel and per degree.** The reordered layout
   helped the operator and the Schwarz application at every degree measured,
   and block-Jacobi not at all (0.9 to 1.04 times).
9. **A block of 32 lanes per block row wastes most lanes.** The hp-multigrid
   smoother gave one warp to each `b x b` block row, with only `b` lanes at
   work (7 of 32 at degree 6). One thread per scalar row made the solve 1.4
   to 2.5 times faster with identical iterations.
10. **Many small library calls are limited by their launches.** In AMGX
    FGMRES with DILU, 58% of the GPU time went to BLAS calls of about 3
    microseconds, one per Krylov basis vector. The same iteration cost
    1.18 ms on a GPU with full-rate FP64 and 1.23 ms on one where FP64 is 64
    times slower than FP32.
11. **The coarse solve can set the cost of an iteration.** With dense LU on
    the coarsest multigrid level, the size of that level set the time per
    iteration: a case with *more* iterations was faster, because its
    coarsest level had 2,690 rows where the other had 5,186.

## Using AMGX

12. **Check which block sizes a preconditioner supports.** The stock DILU
    has no kernels for block sizes 6 and 9, which are the trace blocks of
    degrees 5 and 8, and aborts there. Plain L1-Jacobi needs scalar rows.
13. **Pass block values with their block shape.** The Python binding counts
    blocks from the length of the value array. A flattened array of
    `nnz * b * b` numbers is read as that many blocks and overruns the copy.
    Go through `hybridge.linalg.amgx`, which does this correctly.
14. **Borrow, do not copy.** A device matrix in canonical CSR, or in BSR with
    square blocks, can be handed to the AMGX fork without a copy.

## Timing a solver

15. **Report five times separately:** the setup of the preconditioner, the
    cold solve (setup and first solve), the repeated solve, the warm solves
    for each size of perturbation, and the cost amortized over a sequence,
    each with its iterations.
