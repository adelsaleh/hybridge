# Stationary ADR: ASM + polynomial GMRES, AMGX BSR, and native hp-BSR

This study compares solvers on the same condensed ADR systems, including
99,458 triangles at degree 6 (1,041,187 free trace unknowns). It follows the
[branch qualification](adr_machine_baseline_2026_09_17.md) and the
[assembler comparison](adr_assembler_comparison_2026_09_17.md).
All results describe this machine, these manufactured stationary problems,
and the tested configurations. They are not claims of global solver optimality.

AMGX classical block-AMG and native hp-BSR are substantially faster than the tested one-level ASM+PP configurations for high diffusion. Block-AMG also leads the anisotropic case. The transport case is sensitive to the ASM GMRES restart trigger; see the separate tolerance diagnostic below.

All rows below use the final common contract. Hot **means** use six observations; setup, fresh setup-plus-first-solve, and three-solve amortized costs are **medians**, in milliseconds. Amortization includes algebraic setup, not the common assembly.

| Case | Candidate | Mean hot ms | Iterations | Setup ms | Fresh setup + solve ms | Amortized ms / RHS (3) |
|---|---|---:|---:|---:|---:|---:|
| High diffusion | ASM+PP d=24 | 3467.58 | 150 | 1128.53 | 4597.39 | 3845.41 |
| High diffusion | AMGX BSR FGMRES + block-AMG / Jacobi | 258.50 | 51 | 453.62 | 716.29 | 411.40 |
| High diffusion | Native hp-BSR + GMRES | 254.76 | 66 | 1243.58 | 1498.53 | 669.29 |
| Transport dominated | ASM+PP d=24 | 1414.27 | 62 | 1141.41 | 2555.50 | 1794.82 |
| Transport dominated | AMGX BSR PBICGSTAB + DILU | 637.61 | 263 | 304.09 | 942.55 | 739.17 |
| Transport dominated | AMGX BSR FGMRES + block-AMG / DILU | 537.28 | 35 | 453.27 | 993.65 | 689.99 |
| Anisotropic | ASM+PP d=48 | 6606.60 | 150 | 1201.42 | 7806.24 | 7009.63 |
| Anisotropic | AMGX BSR FGMRES + block-AMG / Jacobi | 646.55 | 117 | 466.16 | 1124.59 | 805.89 |

All 12 final candidates passed all measured and warmup solves. Native hp-BSR also passed. The native and AMGX high-diffusion hot times are too close to establish a clear speed winner from this sample, while AMGX has lower setup cost. For transport, block-DILU wins setup-plus-first-solve, and block-AMG wins hot/amortized solves. “Best” means best among the recorded configurations under the stated cost metric.

| Case | Shared assembly median ms | Space construction ms |
|---|---:|---:|
| High diffusion | 19364.21 | 461.21 |
| Transport dominated | 19704.05 | 457.99 |
| Anisotropic | 19263.39 | 449.87 |

Assembly statistics use one warmup and three repetitions. They include local elimination and host face-system construction; algebraic solver setup and the shared DG-space construction are reported separately. Summing stages is an estimate, not a separately measured end-to-end pipeline.

## Problem and algebra

The domain is `[-1,1]^2`; the exact solution is `sin(pi*x)*sin(pi*y)`.
Conservative transport has velocity `(1,0.5)`, reaction is 1, and the source
is manufactured from the complete ADR operator. The diffusion cases are:

- high diffusion: identity tensor;
- transport dominated: `1e-3 * I`;
- anisotropic diffusion: `diag(1,0.01)`.

The small screening mesh has 2,048 triangles and 21,056 free trace unknowns.
The large mesh has 223 cells in each direction, split into two triangles each.
Its grid-edge length is `2/223`; using triangle diameter in
`Pe = |beta|*h/(2*kappa)` gives approximately 7.09 for the low-diffusion case.
The domain-scale Peclet number is approximately 1,118. The anisotropic case
has a diffusion ratio of 100. These are smooth manufactured solutions; this
study does not test unresolved boundary layers, discontinuous transport, or
transient integration.

Every solver receives the same FP64 Bernstein face matrix and RHS, with
identical elimination, numbering and scaling. AMGX uses natural 7x7 BSR
blocks where supported; CSR candidates are included in the sweep. The
branch's CuPy assembler supplies the immutable shared matrix caches.
The prior assembler study established congruence with master's assembler
when trace coordinates, quadrature and stabilization are matched. Its
performance recommendation is bounded to the measured meshes/orders:
master's fused Numba path for scalar diffusion, and the branch's CuPy path
for tensor diffusion. This study does not extrapolate those assembly
rankings to every problem or replace production defaults.

The measured Frobenius ratios `||A-A.T||_F/||A||_F` are:

| Case | Relative nonsymmetry |
|---|---:|
| High diffusion | 0.000300561 |
| Transport dominated | 0.263635 |
| Anisotropic | 0.000782004 |

The low-diffusion case is strongly nonsymmetric (ratio 0.264); the high-diffusion and anisotropic cases remain nonsymmetric but are more diffusion dominated in this global norm.

## Selection, timing and residual contract

The corrected small-mesh sweep tried 29 candidates per case: 12 ASM+PP
configurations, 7 CSR AMGX configurations and 10 BSR AMGX configurations.
It recorded 65 passes and 22 failures across 87 candidates. Natural-block
aggregation failed with `LOW_DEG not implemented for this block size` in
this installed AMGX build. Classical block-AMG is a separate supported path.
The first large confirmation tried 14 finalists; a further bounded sweep
tried polynomial degrees 12, 24, 36 and 48 with fused applications and seven
AMGX configurations per case. Unsupported, divergent and iteration-limited
runs remain in the evidence and are excluded from ranking.

The original large tests enforced an independently evaluated physical
relative residual of `1e-11`. Several AMGX runs reported internal success
but narrowly failed that physical check. Tightening AMGX's internal target
to `1e-12` did not consistently resolve the problem. A deterministic sample
of 4,096 face rows showed a `4.36e-12` change, relative to the sampled RHS,
between float64 and extended-precision accumulation. This supports a
material roundoff contribution near the original threshold; it is not a
proof of an exact lower bound for every solver.

The final comparison therefore starts a separately recorded protocol with
physical residual `<=1e-10` and internal tolerance `5e-11` for **every**
solver, including native hp-BSR. Earlier strict failures retain their
original status. Candidate selection may retain a near-threshold strict
failure for a new run; it never retroactively marks it passed.

GMRES/FGMRES restart is 75, maximum iterations 1,000, and ASM uses CGS2,
FP64 cuBLAS patch inverses and deterministic polynomial setup. Every solve
starts from zero. In the installed AMGX stack, the `zero_initial_guess`
flag alone does not clear a reused solution vector. The harness explicitly
calls `set_zero` before every timed solve and includes its cost. This fix
invalidates the earlier CSR-only screen for ranking; the corrected BSR
screen and subsequent studies use explicit resets. BSR downloads use a
full scalar-length destination buffer.

Each final candidate uses one warmup setup with three solves, followed by
three fresh setups with three solves apiece. Setup includes the candidate's
matrix conversion/upload, preconditioner construction and reusable solver
state. Hot times use the second and third solve of each measured setup
(six observations). Fresh setup-plus-solve and amortized cost over three
solves are separate statistics. GPU work is synchronized. Independent host
residual evaluation, local reconstruction and PDE-error integration occur
outside the solve timer. All GPU jobs run serially. Diagnostic profiling is
separate and never used to rank candidates.

Native hp-BSR uses master's existing hp hierarchy and scalar AMGX p=0
coarse cycle on `(A+A.T)/2` in orthonormal Legendre coordinates. A reusable
adapter transforms residuals/corrections while outer GMRES continues to
apply the original nonsymmetric Bernstein matrix. This does not run PCGF
on a nonsymmetric ADR operator. Both standard and robust hp policies were
screened; the standard policy was retained for the final high-diffusion run.

### Applications and separate component timings

Application times are diagnostic CUDA-event means. ASM/native columns use deterministic probe vectors after warmup; AMGX uses its existing timers on actual Krylov preconditioner calls, with synchronization per call. They explain costs but are not interchangeable with uninstrumented wall times. Counts are per zero-guess solve.

| Case | Candidate | Complete preconditioner ms / call | Preconditioner calls | Outer operator calls | Total fine operator calls incl. PP | ASM calls |
|---|---|---:|---:|---:|---:|---:|
| High diffusion | ASM+PP d=24 | 21.074 | 152 | 153 | 3801 | 3800 |
| High diffusion | AMGX BSR FGMRES + block-AMG / Jacobi | 4.128 | 51 | not exposed | not exposed | not exposed |
| High diffusion | Native hp-BSR + GMRES | 2.584 | 68 | 69 | level counts not exposed | — |
| Transport dominated | ASM+PP d=24 | 20.972 | 64 | 65 | 1601 | 1600 |
| Transport dominated | AMGX BSR PBICGSTAB + DILU | 0.845 | 525 | not exposed | not exposed | not exposed |
| Transport dominated | AMGX BSR FGMRES + block-AMG / DILU | 14.506 | 35 | not exposed | not exposed | not exposed |
| Anisotropic | ASM+PP d=48 | 41.613 | 152 | 153 | 7449 | 7448 |
| Anisotropic | AMGX BSR FGMRES + block-AMG / Jacobi | 4.383 | 117 | not exposed | not exposed | not exposed |

AMGX application counts come from immediate-preconditioner native timing callbacks, including the early final PBICGSTAB exit (525 applications for 263 iterations). Exact AMGX outer/internal multilevel SpMV counts are not exposed by the installed bindings and are explicitly unavailable. Native hp outer counts exclude its internal hierarchy operations. No native rebuild was performed to add counters.

In the instrumented ASM solves, preconditioning accounts for about 92.5%, 95.2% and 96.0% of GPU operation time for high diffusion, transport dominance and anisotropy. A single fused ASM call is about 0.415 ms on the high-diffusion system; polynomial repetition multiplies this cost.

### Tolerance-triggered restarts

A separate repeated transport run keeps the physical acceptance limit at `1e-10` but tightens only ASM's internal tolerance to `1e-11`. Degree 24 then takes **730.37 ms mean hot time / 32 iterations**, versus 1,414.27 ms / 62 at the common internal `5e-11`. This sensitivity result uses a different internal trigger and is not silently substituted into the common-tolerance table.

The diagnostic reuses exactly the same operator and polynomial-preconditioner object. At `5e-11`, the first cycle ends after 31 iterations with physical relative residual `1.71468e-10`, then another 31-iteration cycle oversolves to about `6.11e-14`. At `1e-11`, one 32-iteration cycle reaches `7.94e-12`. The current code compares the projected residual against `rtol * beta` each cycle and restarts after a failed physical check. Retaining the existing Krylov basis or adapting the cycle trigger is a concrete optimization follow-up; this study does not change that algorithm.

## Validation and memory

The preceding branch baseline passed 37 ADR tests, 30 CPU validation cases,
6 GPU validation cases and all 190 performance-smoke jobs. On the screening
mesh, the shared CuPy assembly was checked against independent NumPy
assembly and a CPU sparse-direct solution. At the large size, validation
uses the original host face operator, reconstructed manufactured-solution
L2 error and cross-solver agreement; it does not claim a large CPU-direct
reference.

The final audit covers **144 solves** (12 candidates, three measured setups plus one warmup, three solves per setup). All pass; the worst physical relative residual is `4.91554e-11`. Native hp passes its additional 12 solves. Each case has one shared operator/RHS SHA256 across all final ASM and AMGX candidates.

| Case | Reconstructed L2-error range across final candidates | Maximum relative trace difference vs a passing ASM solution |
|---|---:|---:|
| High diffusion | 1.276e-12–1.376e-12 | 3.456e-13 |
| Transport dominated | 3.410e-14–6.579e-12 | 6.627e-12 |
| Anisotropic | 1.292e-12–2.388e-12 | 2.548e-12 |

Native hp is included in the high-diffusion trace-agreement check. These errors are already close to floating-point/local-assembly accuracy for this smooth p=6 solution; no asymptotic convergence-rate claim is made. All 21 separate profiling/diagnostic jobs completed successfully. The full master documentation gate has 6 passes and 3 failures in unrelated existing directory/link/docstring checks; focused checks of the new ADR links and adapter docstrings pass.

| Case | Candidate | CuPy tracked peak GiB | AMGX managed peak GiB | Sampled device increase GiB | Peak host RSS GiB |
|---|---|---:|---:|---:|---:|
| High diffusion | ASM+PP d=24 | 2.247 | — | 2.928 | 9.027 |
| High diffusion | AMGX BSR FGMRES + block-AMG / Jacobi | 0.000 | 2.663 | 2.975 | 9.010 |
| High diffusion | Native hp-BSR + GMRES | 1.358 | 0.093 | 1.749 | 9.203 |
| Transport dominated | ASM+PP d=24 | 2.247 | — | 2.928 | 9.027 |
| Transport dominated | AMGX BSR PBICGSTAB + DILU | 0.000 | 0.451 | 0.605 | 8.779 |
| Transport dominated | AMGX BSR FGMRES + block-AMG / DILU | 0.000 | 2.552 | 2.842 | 8.999 |
| Anisotropic | ASM+PP d=48 | 2.247 | — | 3.309 | 9.027 |
| Anisotropic | AMGX BSR FGMRES + block-AMG / Jacobi | 0.000 | 2.612 | 2.920 | 8.998 |

The shared CuPy assembly measured approximately **18.504 GiB** of hook-tracked peak live allocations, **18.545–18.548 GiB** sampled device-memory increase, and **27.752–27.758 GiB** peak host RSS across the three cases. Assembly profiling used one separately instrumented assembly per case; solver profiling used one setup/two solves.

CuPy memory figures track allocator hooks; AMGX figures come from its
managed allocator. The whole-device sample is taken at a target 10 ms
interval and is a lower bound on the peak; it includes library allocations
and other processes. The increase subtracts the process's measured
post-context baseline. Host RSS includes cached arrays and validation.
Assembly and solver memory are profiled separately. These distinctions
prevent allocator snapshots from being mistaken for exact whole-device
peak memory.

## Environment and reproducibility

The GPU is an RTX PRO 5000 Blackwell, driver 580.126.09. The environment
uses Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, Numba 0.67.0 and CuPy 14.2.0.
CuPy reports runtime 13.2; the AMGX guard selects the existing CUDA 13.0
native build. BLAS/OpenMP thread counts are one. The branch's strict lock
preflight rejects version/toolkit-package differences, while the actual
CPU/GPU tests pass; see the baseline report. Runtime CUDA/Numba JIT was
explicitly authorized. No native build or installation was performed.

The experiment worktree is `gpu_gmres_precondit` at
`d44acce873b18daa6507e0c89b20a9e4e5ab913e`. Master remains a separate,
previously dirty checkout. The native AMGX and binding source trees are
also dirty, so their commit IDs alone cannot identify the binaries.
The actual loaded build-library SHA256 is
`f19780cea8929352c8d3cf97ed2a2f43a23180a9932e9ba95a0af4cb1b056023`;
the installed pyamgx extension SHA256 is
`0fa650106c352469750088d2594b4e66c2baf6a1824252fee2364db327fc2841`.
The unused installed AMGX library differs from the library loaded from the
build directory. Binary paths, hashes, repository states, worker hashes,
exact configurations, measurements and failures are retained in
[`adr_solver_comparison_2026_09_17.data.json`](adr_solver_comparison_2026_09_17.data.json).
Raw local artifacts are under `/home/adelsaleh/src/hdgfem-gmres/run_logs/`.

Use new output directories when replaying the commands below. Each Python
command runs through the existing guard; this shell function does not build
anything:

```bash
cd /home/adelsaleh/src/hdgfem-gmres
source run_logs/adr_baseline_20260917/environment.sh
adr_python() {
  HDGFEM_CUDA13_ROOT=/usr/local/cuda-13.0 \
  HDGFEM_AMGX_BUILD_ROOT=/home/adelsaleh/src/AMGX-build-cuda13 \
  HDGFEM_AMGX_INSTALL_ROOT=/home/adelsaleh/src/AMGX-install-cuda13 \
    /home/adelsaleh/src/hybridge/scripts/gpu/run_cuda13.sh python "$@"
}
adr_python scripts/run_adr_solver_comparison.py --degree 6 --mesh 32 \
  --output run_logs/new_screen
adr_python scripts/run_adr_solver_comparison.py --degree 6 --mesh 223 \
  --select-from run_logs/new_screen --timeout 1800 \
  --output run_logs/new_large
adr_python scripts/refine_adr_solver_comparison.py \
  --source run_logs/new_large --output run_logs/new_tuning
adr_python scripts/refine_adr_solver_comparison.py \
  --source run_logs/new_large --confirm-from run_logs/new_tuning \
  --rtol 1e-10 --internal-rtol 5e-11 --allow-near-threshold \
  --output run_logs/new_final
```

A strict phase can exit nonzero because a family has no passing candidate;
that is a recorded numerical result, not permission to ignore its failures.
Native and profiling runs use the recorded JSON specifications with
`scripts.adr_native_hp_worker`, `scripts.profile_adr_solver_memory`,
`scripts.profile_adr_amgx_preconditioner` and the canonical
`scripts.adr_performance_worker` component profiler. Worker snapshots retain
older protocols before harness changes. The new native adapter is
`hdgfem/linalg/face_hp_krylov.py` in the master worktree; it is experimental
and does not change the production solver selection.
