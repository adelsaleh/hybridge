# Kernel tuning by GPU architecture

This page gathers, for each GPU architecture on which HYBRIDGE was measured,
the kernel settings that were fastest there, and the lessons that the
measurements taught. It is a summary. The numbers, meshes and commands are in
the study of each architecture, and the other pages of this folder are listed
in its [index](README.md):

| Compute capability | Measured on | Study |
|---|---|---|
| 7.5 (Turing) | Quadro RTX 6000, 24 GB | [`quadro_rtx6000_sm75`](../algorithms/assembly/quadro_rtx6000_sm75/README.md) |
| 9.0 (Hopper) | H200 NVL, 144 GB | [`h200_sm90`](../algorithms/assembly/h200_sm90/README.md), [solver kernels](../../campaigns/solver_h200/README.md) |
| 12.0 (Blackwell) | RTX PRO 5000, 48 GB | [`rtx_pro_5000_sm120`](../algorithms/assembly/rtx_pro_5000_sm120/README.md) |

A setting listed here is a starting point, not a guarantee. On first use the
tuner (`hybridge/runtime/cuda_tuning.py`) times at most eight equivalent
candidates within 30 seconds on a sample of the actual problem and keeps the
fastest one that reproduces a reference result. What this page changes is the
order in which candidates are tried, and which settings are worth giving
explicitly. [Architecture tuning](../development/architecture_tuning.md)
describes the tuner and its controls.

Notation: `p` is the polynomial degree, `NEL = (p+1)(p+2)/2` the number of
unknowns of one scalar field on a triangle (55 at `p = 9`), and a *launch* is
the number of threads per block with which a kernel is started. AR is
advection-reaction, DR diffusion-reaction (Poisson), ADR
advection-diffusion-reaction. A kernel is *resident* on a streaming
multiprocessor (SM) when its registers and shared memory fit there; the
number of resident blocks per SM is what the text calls occupancy.

## 1. The properties that decide

Four properties of the device explain almost every difference below.

| Property | Capability 7.5 | Capability 9.0 | Capability 12.0 |
|---|---|---|---|
| FP64 speed relative to FP32 | not recorded | 1/2, with FP64 tensor cores | 1/64 |
| Shared memory one block may request | not recorded | 227 KB | 99 KiB |
| SMs on the measured GPU | not recorded | 132 | 110 |
| Registers of the fused AR kernel at `p = 7..9` | not recorded | 254 to 255 | 80 to 88 |

- **FP64 speed** decides whether dense local algebra is limited by
  arithmetic. On capability 12.0 it is; on capability 9.0 it is not, and the
  kernels are limited by something else (see the lessons).
- **Shared memory per block** decides whether the work arrays of one element
  fit in one block, and how many such blocks an SM can hold.
- **Registers per thread** decide how many warps an SM can hold. The same
  source compiles to very different register counts on different
  architectures, so the compiled count must be read back, not assumed.
- **Memory bandwidth** decides the solver kernels, which read much and
  compute little (about 0.25 floating-point operations per byte).

## 2. Settings per architecture

"Automatic" means that `"auto"` already chooses this value on that
architecture. "Not measured" means that no record exists; the tuner then
starts from the conservative baseline.

### 2.1 Compute capability 9.0

| Family | Setting | Value | Measured effect |
|---|---|---|---|
| AR | assembly route | `raw_local_assembly="tensor"` from `p = 5` (`auto` times it against the other routes when the advection field is in the solution space, and it won from `p = 5` here) | 2 to 4 times faster than the single-kernel and three-kernel routes at `p = 6..9` |
| AR | local solve of the tensor route | register Gauss–Jordan (`fused`), found by the tuner | 1.3 to 3.4 times faster than MAGMA at `p = 6..9` |
| AR | chunk of the tensor route | 16 GiB of work arrays (default) | 2 million triangles at `p = 9` in 17.6 GB instead of 128 GB |
| AR | volume term of the three-kernel route | `inline` (default) | as one matrix product the build step takes 26 ms instead of 54 at `p = 9`, but the tensor route is faster still |
| DR | assembly launch | all right-hand-side columns in one batch; 32 threads up to `NEL = 15`, 64 at 21, 128 up to 45, 256 at 55 (automatic candidate order) | the earlier order was 1.1 to 4 times slower when only four candidates were timed |
| DR | stored local factors | `cache_local_factors="schur-lu"`, or `"schur-lu+responses"` when the memory allows it | responses: reconstruction 1.4 to 3.3 times faster, a time step 0.2 to 3% faster, 1.5 to 2.5 times the stored memory |
| DR | Cholesky factors | not recommended | faster, but the potential differs from the LU result by 5e-10 relative |
| ADR | launch for variable tensors | 256 threads at `p = 8`, 512 at `p = 9` (automatic) | time halves with each doubling while one block fits per SM |
| ADR | variable full-tensor mass | explicit inverse from `p = 7` (default) | assembly 1.8 to 3.1 times faster at `p = 7..9` |
| ADR | constant coefficients | 128 threads (default) | wider launches are slower |
| Solvers | face-dense operator and Schwarz | `implementation="interleaved"`, `application="interleaved"` | 1.8 and 2.2 times faster at `p = 6`; repeated solves 1.6 to 2 times faster |
| Solvers | block-Jacobi | default layout | the other layouts give 0.9 to 1.04 times |
| Solvers | hp-multigrid | `kernels="packed"` or `"packed-fused"` | 1.4 to 2.5 times faster, same iterations |
| Solvers | GMRES after a warm start | `estimate_target="true_target"` | 58 to 33 iterations on one Schwarz system |

### 2.2 Compute capability 12.0

| Family | Setting | Value | Measured effect |
|---|---|---|---|
| AR | assembly route | three-kernel route (`split3`) at `p = 5..7`, tensor route with MAGMA at `p = 8..9` | at `p = 9`, 117,439 triangles: single kernel 699 ms, three kernels 389 ms, tensor 299 ms |
| AR | launch of the single-kernel route | 32 threads up to `p = 6`, 64 at `p = 7`, 128 at `p = 8..9` | from the sweeps on capabilities 7.5 and 12.0 |
| DR | kernel | raw `poisson_local` with the generic candidate order (coupling kept in shared memory when FP32 is at least 16 times faster than FP64) | 1.4 to 4 times faster than the CuPy assembly |
| ADR | launch | 32 threads up to `p = 2`, 64 up to `p = 4`, 128 up to `p = 9` | from the sweeps on capabilities 7.5 and 12.0 |
| ADR | variable tensors at `p >= 7` | mass factor in global memory (automatic) | the 99 KiB limit leaves no room for it in shared memory |
| ADR | timings at `p >= 7` | not measured | the kernels are qualified (508 tests) but not timed |
| Solvers | all kernel variants of 2.1 | not measured | |

The settings of 2.1 that depend on full-rate FP64 are not used here: the
register Gauss–Jordan solve, the chunked tensor route as the default from
`p = 5`, the explicit ADR mass inverse with the mass in shared memory and the
Poisson candidate order are restricted to capability 9.0 or to devices where
FP32 is at most 4 times faster than FP64.

### 2.3 Compute capability 7.5

| Family | Setting | Value | Measured effect |
|---|---|---|---|
| AR | launch of the single-kernel route | 32 threads up to `p = 6`, 64 at `p = 7`, 128 at `p = 8..9` | origin of this table; `p = 8..9` checked again with CUDA 13 |
| ADR | launch | 32 threads up to `p = 2`, 64 up to `p = 4`, 128 up to `p = 9` | origin of this table |
| AR tensor route, register solve, chunking, ADR mass inverse | | not measured | |
| Solver kernel variants | | not measured | |

### 2.4 One problem on two architectures

The same problems run 4 to 15 times faster on the capability 9.0 GPU than on
the capability 12.0 GPU, although its FP64 arithmetic is about 40 times
faster: AR raw kernels 4 to 5 times, AR tensor route 8 to 13 times, Poisson
assembly 2.6 times at `p = 2` and 7.9 to 9.4 times at `p = 9`, the ADR tensor
kernel 14 to 15 times at `p = 6`. Native solver kernels were 1.2 to 2.7 times
faster before their memory layout was changed, and 4.4 times after. The gap
between 40 and these factors is the subject of the lessons.

## 3. Lessons

Each lesson names the record it comes from.

### 3.1 What limits a kernel

1. **Small factorizations and triangular solves are limited by waiting, not
   by arithmetic.** Each elimination step ends in a barrier, and each pivot
   search and substitution depends on the previous one. On capability 9.0
   the Poisson kernel ran at about 20% of the FP64 peak and below 1.5% of
   the memory bandwidth. Rewriting the solve helped where faster arithmetic
   could not: the register Gauss–Jordan solve is 1.3 to 3.4 times faster
   than MAGMA, and solving each local column on one warp took the Poisson
   right-hand side at `p = 9` from 13.6 to 7.2 ms.
   ([`h200_sm90`](../algorithms/assembly/h200_sm90/README.md) 2.5 and 3.3,
   [Poisson local solves](poisson_local_solves.md) 2 and 3)
2. **A high-order kernel with one element per block runs out of resident
   blocks.** At `p = 9` the ADR kernel needs 140 KiB of shared memory, so
   one block of four warps fits per SM. Adding blocks is impossible, and
   giving that block more threads helps: the assembly takes 2.8 s with 128
   threads and 1.6 s with 256. ([`h200_sm90`](../algorithms/assembly/h200_sm90/README.md) 4.1 and 4.3)
3. **The total shared memory per SM has a step.** With eight blocks of 64
   threads, raising the total from 176 to 192 KB makes the Poisson
   right-hand side at `p = 8` go from 7.1 to 10.4 ms. The `p = 9` kernel
   needs 203 KB for eight blocks, so the tuner picks four blocks of 128
   threads. The cause, a cache that shrinks as shared memory grows, is
   inferred and was not measured.
   ([Poisson local solves](poisson_local_solves.md) 2)
4. **Read the register count of the compiled kernel.** The single-kernel AR
   assembly with conflict-averaged upwind compiles to 254 or 255 registers
   on capabilities 8.0 and 9.0, which leaves 8 warps per SM at 128 threads,
   and to 80 to 88 registers on capabilities 10.0 and 12.0. Plain upwind
   needs 82 to 92. ([`h200_sm90`](../algorithms/assembly/h200_sm90/README.md) 2.1)
5. **Dense algebra with no reuse between elements belongs in one matrix
   product over all elements.** That form runs on the FP64 tensor cores. On
   capability 9.0 it makes the AR tensor route the fastest from `p = 5`. On
   capability 12.0, where FP64 is slow, it wins only at `p = 8..9`.
   ([assembly](../algorithms/assembly/README.md) 5)
6. **Solver kernels are limited by how they read memory.** The face-dense
   operator and the Schwarz application read 21 to 35 numbers per row with
   one thread per row, and reached 30 to 42% of the bandwidth. Storing the
   matrices in the order the threads read them reached 67 to 76%. The
   launch changed these kernels by a few percent at most.
   ([solver kernels](../../campaigns/solver_h200/README.md))
7. **A Krylov solver with many small kernels is limited by launches.** With
   AMGX FGMRES and DILU, 58% of the GPU time was spent in BLAS calls of
   about 3 microseconds each, one per basis vector. An iteration cost the
   same on capabilities 9.0 and 12.0.
   ([solver kernels](../../campaigns/solver_h200/README.md))

### 3.2 What to store and what to recompute

8. **Storing factors is most of the gain.** Without stored LU factors a
   Poisson time step was 4 to 28% slower, and reconstruction at `p = 9` took
   28.7 ms against 4.65 ms.
   ([Poisson local solves](poisson_local_solves.md) 4)
9. **Keep what assembly already computed.** Assembly solves every trace
   column, and the stored-LU option used to discard those solutions. Keeping
   them removes the local solve from reconstruction (4.65 to 1.56 ms at
   `p = 9`) at no setup cost. (same record)
10. **An explicit inverse pays when it is applied many times per element.**
    The ADR kernel applies the inverse mass matrix to every column of the
    local system, and is 1.4 to 2 times faster with it. The Poisson
    kernel applies its local inverse once per right-hand side: the gain was
    0.2 to 1.2% of a time step, for 33 to 69 ms of setup per operator and a
    residual of 1e-13 where a solve gives 1e-15. It was not adopted.
    ([`h200_sm90`](../algorithms/assembly/h200_sm90/README.md) 4.5,
    [Poisson local solves](poisson_local_solves.md) 4)
11. **Reordering a product without removing work gains nothing.**
    Contracting with the reference tables first, loading the factor with
    several reads per thread and the compiler level `-O1` left the `p = 9`
    Poisson kernels unchanged or slower, and a register cap of 80 gained
    less than the warp solve that replaced it. They are kept as switches
    that are off.
    ([Poisson local solves](poisson_local_solves.md) 3)

### 3.3 The cost around a kernel

12. **Bookkeeping on the host can cost more than the kernel.** Reading the
    device properties takes 0.7 to 0.9 ms, and one Poisson launch did it
    three times: 2.8 of the 3.4 ms of a right-hand side on a small mesh,
    where the kernel itself took 0.04 ms. Reading them once per process and
    choosing the launch once per operator made a time step up to 34%
    faster. The same lookup cost 4.4 ms per ADR assembly.
    ([Poisson local solves](poisson_local_solves.md) 1)
13. **Give every lookup a time limit relative to the kernel it serves.** A
    lookup of 2 ms is harmless before an assembly and dominant before a
    kernel of 10 microseconds. The first form of the measured launches for
    the recovery, averaging and gradient kernels made time steps a median
    9.7% slower for this reason alone; its second form, which looks up once
    per process, costs 7 microseconds per call.
14. **Measuring the launch of a small kernel rarely pays.** For those three
    kernels, 76 of 87 measurements chose 256 threads, the best alternative
    gained at most 4.7% on a kernel that takes at most 1.6% of a time step,
    and the time step did not change.
15. **The order of the candidates matters when only a few are timed.**
    Launch orders learned where FP64 is slow put the wrong candidates first
    where it is fast: with four candidates the Poisson tuner was 1.1 to 4
    times slower than an exhaustive search. A stored choice is also tied to
    the mesh-size class and to whether the boundary data are zero, so a
    record from another mesh size is not reused.
    ([`h200_sm90`](../algorithms/assembly/h200_sm90/README.md) 3.2)
16. **A solver that starts from a good guess must stop at the requested
    residual.** A stopping test relative to the residual at the start of
    each restart cycle made a warm-started solve iterate to round-off: 57
    iterations for every size of the perturbation. `estimate_target=
    "true_target"` removes this. It leaves less margin on the error, and
    some high-degree runs then met the residual target but missed the bound
    on the L2 error. ([solver kernels](../../campaigns/solver_h200/README.md),
    [solvers](solvers.md) 5)

### 3.4 Correctness

17. **Pivoting is rare and still necessary.** Only 0.2 to 0.3% of the local
    AR matrices exchange a row, but without pivoting the forward error is 25
    times larger. ([`h200_sm90`](../algorithms/assembly/h200_sm90/README.md) 2.5)
18. **Two correct factorizations can give different answers.** The Cholesky
    and the pivoted LU factorization of the same local Poisson matrix differ
    by 5e-10 relative in the potential. Which one is closer to the exact
    local solution was not established, so the Cholesky option is not the
    default. ([Poisson local solves](poisson_local_solves.md) 4)
19. **A timing in FP32 cannot choose FP64 arithmetic,** and a tuner must not
    change precision, quadrature or tolerances.
    ([architecture tuning](../development/architecture_tuning.md))
20. **Keep coefficients on the device.** Coefficients that only existed as
    NumPy functions were sampled on the host: 67.6 s of a 74.7 s assembly.
    On a smaller mesh, sampled on the device, they took 77 ms.
    (ADR mesh-ladder campaign, `campaigns/adr_h200_tuned`)

### 3.5 Measuring

21. **Compare in pairs and repeat.** One repetition cannot show a change.
    On a shared node, a call into the driver stalled for up to 165 ms about
    every 36 s, and a short run can have such a stall in its median.
    (ADR selection-cost study, `campaigns/adr_selection_cost`)
22. **Vary one thing.** A table can suggest a cause; only a rerun in which a
    single parameter changes can support it.
23. **Read the numbers, not only the pass marks.** A first run passed every
    check while it was 30% slower with no kernel gain.

Lessons 13, 14, 22 and 23 come from the kernel-tuning campaign of the
research repository (`guiding-center-cg-hdg`, campaign
`gc_article_kernel_tuning`).

## 4. A new architecture

1. Run the solves with the defaults. The tuner starts from the conservative
   baseline and measures.
2. Read the registers and shared memory of the compiled kernels from the
   tuning records, and compare them with section 1.
3. Run the qualification of
   [`campaigns/architecture_tuning`](../../campaigns/architecture_tuning/README.md).
4. Time representative degrees and mesh sizes with the defaults and with the
   settings of the closest architecture in section 2.
5. Add a folder under `docs/algorithms/assembly/` with the platform, the
   results per family, the defaults and what was not measured, and add the
   architecture to this page.
