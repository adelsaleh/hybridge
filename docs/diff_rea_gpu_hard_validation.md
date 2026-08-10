# Hard-case GPU GMRES validation

This campaign validates the production face-dense GPU solver on every problem
registered in `scripts/diff_rea_cases.py`, with particular emphasis on the
trigonometric Poisson problem and the variable anisotropic tensor-diffusion
problem.  It is designed for float64 tolerances of `1e-11` to `1e-12` and for
polynomial orders `p=4,5,6`.

It produces three separate deliverables:

1. raw repeated timings and component/Nsight profiling data;
2. exhaustive numerical PASS/FAIL validation for all registered cases;
3. a consolidated analysis of kernels, polynomial degrees, and preconditioner
   combinations, restricted to numerically valid candidates.

## What is now covered

The production API accepts all six requested preconditioner families:

| Campaign name | Production API name | Action |
| --- | --- | --- |
| `none` | `none` | Unpreconditioned GMRES |
| `polynomial` | `poly` | Harmonic-Ritz/Leja polynomial |
| `bj` | `block_jacobi` | Face Block-Jacobi |
| `bj-polynomial` | `block_jacobi_poly` | Polynomial in the BJ-preconditioned operator |
| `asm` | `asm` | Element-patch additive Schwarz |
| `asm-polynomial` | `asm_poly` | Polynomial in the ASM-preconditioned operator |

The validator can independently sweep:

- face-dense matvec: `raw`, `raw_fused`, `matmul`, or `auto`;
- ASM action: `raw`, `fused`, `matmul`, or `auto`;
- Block-Jacobi action: `raw`, `matmul`, or `auto`;
- polynomial degree;
- GMRES orthogonalization: `cgs`, `cgs2`, `mgs`, or `mgs2`;
- cases, orders, mesh sizes, restart, tolerance, and repetition count.

`auto` measures `raw`, `raw_fused`, and `matmul` operator paths, all three ASM
paths, and both Block-Jacobi paths on the current GPU. It checks numerical
equivalence before persisting the winner.
Operator-only and operator+ASM tuning records are intentionally kept separate,
so an operator-only cache entry cannot suppress ASM tuning later.

## Validation gates

A measured solve is marked as passed only when all of the following hold:

1. GMRES reports convergence.
2. All trace and error metrics are finite.
3. The recomputed true relative residual is at most `5 * rtol` by default.
4. The physical condensed-system relative residual is at most `10 * rtol`.
5. The full trace agrees within `1e-8` relative norm with an independently
   assembled SciPy direct solve for the same case/order.

The report also stores primal and flux L2 errors, every operation count,
restart-cycle and CGS-to-CGS2 fallback counts, Arnoldi orthogonality defects,
workspace sizes, and requested versus autotuned kernel choices.

The exhaustive validation uses `--reference-solver direct`. Performance-only
sweeps use `first-passing` after that independent validation to avoid repeatedly
paying for a large CPU direct solve.

The trace agreement threshold is deliberately not set equal to `rtol`.
Residual accuracy and forward-solution accuracy differ by the condition number,
especially for anisotropic diffusion.

## Recommended execution order on a new GPU

Verify the environment first:

```bash
PYTHONPATH=. python scripts/validate_gpu_environment.py
```

Inspect the exact commands without using GPU time:

```bash
PYTHONPATH=. python scripts/run_diff_rea_gpu_hard_campaign.py \
    --campaign full \
    --output results/hard_gpu \
    --dry-run
```

Run the short bring-up campaign:

```bash
PYTHONPATH=. python scripts/run_diff_rea_gpu_hard_campaign.py \
    --campaign smoke \
    --output results/hard_gpu_smoke \
    --continue-on-error
```

Then run the mathematical validation on all seven registered cases, including
`rotated-anisotropic-sine`, at `p=4,5,6`:

```bash
PYTHONPATH=. python scripts/run_diff_rea_gpu_hard_campaign.py \
    --campaign validation \
    --output results/hard_gpu_validation \
    --continue-on-error
```

Finally, run the complete performance search:

```bash
PYTHONPATH=. python scripts/run_diff_rea_gpu_hard_campaign.py \
    --campaign full \
    --output results/hard_gpu_full \
    --continue-on-error
```

The full campaign is intentionally expensive.  In addition to the all-case
validation, it performs repeated kernel and degree sweeps on the trigonometric
and anisotropic cases, followed by component-level CUDA profiling assembled
from those same two difficult PDEs (not from the quadratic Poisson surrogate).

The final stage reads all preceding CSV/JSON files and writes
`final_analysis.md`, `final_analysis.json`, and separate CSV tables for
case/order winners, globally robust configurations, kernel winners,
polynomial-degree winners, GMRES bottlenecks, and primal/flux error trends over
`p=4,5,6`.

The canonical script lives at
`scripts/run_diff_rea_gpu_hard_campaign.py`.  A repository-root launcher with
the same basename is also included, so the shorter command below is equivalent:

```bash
PYTHONPATH=. python run_diff_rea_gpu_hard_campaign.py --campaign smoke
```

## Targeted commands

To test the two difficult cases at `p=4,5,6` with every preconditioner and a
manageable degree set:

```bash
PYTHONPATH=. python scripts/validate_diff_rea_gpu_hard_cases.py \
    --cases trigonometric-poisson tensor-sine \
    --orders 4 5 6 \
    --preconditioners none polynomial bj bj-polynomial asm asm-polynomial \
    --polynomial-degrees 8 12 18 24 \
    --operators auto \
    --asm-applications auto \
    --rtol 1e-12 \
    --restart 100 \
    --max-iterations 5000 \
    --output-prefix results/hard_targeted
```

To compare matvec and ASM kernels without autotuning:

```bash
PYTHONPATH=. python scripts/validate_diff_rea_gpu_hard_cases.py \
    --cases trigonometric-poisson tensor-sine \
    --orders 4 6 \
    --preconditioners asm \
    --operators raw raw_fused matmul \
    --asm-applications raw fused matmul \
    --no-autotune \
    --warmup-solves 1 \
    --repeats 5 \
    --rtol 1e-12 \
    --output-prefix results/kernel_comparison
```

To compare polynomial degrees for the three polynomial families:

```bash
PYTHONPATH=. python scripts/validate_diff_rea_gpu_hard_cases.py \
    --cases trigonometric-poisson tensor-sine \
    --orders 4 5 6 \
    --preconditioners polynomial bj-polynomial asm-polynomial \
    --polynomial-degrees 4 8 12 18 24 32 \
    --operators auto \
    --asm-applications auto \
    --warmup-solves 1 \
    --repeats 5 \
    --rtol 1e-12 \
    --output-prefix results/polynomial_comparison
```

## Outputs and ranking

For an output prefix `results/run`, the validator writes:

- `results/run_raw.csv`: one row per measured repetition;
- `results/run_summary.csv`: medians, worst residuals, pass counts, and ranks;
- `results/run.json`: metadata, raw rows, summaries, and winners per case/order.

Repeated summaries retain minimum, median, mean, standard deviation, and p90
solve time. For each `auto` configuration, the first post-warm-up measured
sample forces a retune, so the cold-start metric contains the real tuning cost
rather than a cache lookup.

Four rankings are retained because they answer different questions:

| Rank | Includes | Use it for |
| --- | --- | --- |
| `solve_rank` | GMRES solve only | Repeated solves with reused setup |
| `hot_time_rank` | operator + preconditioner setup + solve | One solve with cached autotuning |
| `cold_time_rank` | forced retune + setup + solve | First solve on a new architecture/problem size |
| `end_to_end_rank` | complete HDG path | Application-level runtime |

Only configurations whose every repetition passed receive ranks.  A failed
unpreconditioned run is useful information about robustness; it does not by
itself imply a GMRES implementation defect if the preconditioned variants meet
the same tight residual and trace checks.

The campaign uses `--require-coverage`: an individual failed candidate is kept
in the reports without making the command fail, provided another configuration
passes every repetition for the same case/order.  The validator still returns a
nonzero status for execution exceptions or an uncovered case/order.  Use
`--strict` instead only when every requested candidate is required to pass.

## Profiling

The full campaign always runs the built-in component profiler.  It separates
gather, dense face product, Block-Jacobi, ASM restriction/local/prolongation,
polynomial/hybrid applications, and instrumented versus repeated uninstrumented
GMRES costs.

If Nsight tools are installed, add one or both optional captures:

```bash
PYTHONPATH=. python scripts/run_diff_rea_gpu_hard_campaign.py \
    --campaign full \
    --output results/hard_gpu_profiled \
    --continue-on-error \
    --with-nsys \
    --with-ncu
```

Nsight Systems should be used first to identify launch gaps, synchronizations,
and dominant kernels.  Nsight Compute is much slower and is most useful after
Systems has narrowed the investigation to a small number of kernels.

## Practical interpretation

For a single right-hand side, prefer the passing configuration with the best
`hot_time_rank` rather than the smallest iteration count.  A high polynomial
degree can reduce Krylov iterations yet lose overall because each polynomial
application performs more operator/base-preconditioner actions.  For many
right-hand sides on the same matrix, compare `solve_rank` and amortize the
reported preconditioner setup time over the number of solves.

At `rtol=1e-12`, retain float64, direct Dirichlet elimination, true-residual
replacement, and orthogonality monitoring during validation.  After a winner
has proved stable, orthogonality monitoring can be disabled for the final
steady-state timing because forming the diagnostic Gram matrix is intentional
extra work.
