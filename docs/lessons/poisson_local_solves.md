# The cost of one Poisson solve per element

A time step of a coupled transport and Poisson calculation refreshes the
Poisson right-hand side, solves the trace system, and reconstructs the local
fields, several times. This page records where the time of the refresh and
of the reconstruction went on a GPU of compute capability 9.0 (an H200 NVL),
what was tried, and what each attempt gave. Assembly of the operator is not
the subject: it happens once.

All numbers are per Poisson solve in FP64, measured in real time-stepping
runs unless a line says *replay*. A replay launches a kernel alone with the
arguments captured from a real run. `p` is the polynomial degree and `N` the
number of triangles. A change is called *at parity* when the final fields
differ from the unchanged code by no more than two unchanged runs differ from
each other.

## 1. The fixed cost was on the host

On a small mesh (`p = 4`, `N = 1,474`) the refresh took 3.67 ms and the
reconstruction 3.41 ms, and the kernels themselves at most 0.04 ms each.

| Operation | Refresh | Reconstruction |
|---|---:|---:|
| Reading the device properties, three times per launch | 2.84 ms | 2.78 ms |
| Looking up the stored launch (hashing, legality checks, record) | 0.17 | 0.14 |
| Building the candidate list and the kernel arguments | 0.17 | 0.16 |
| Two round trips to the device (is the trace zero, did a factorization fail) | 0.06 | 0.05 |
| Launching the kernel and waiting for it | 0.06 | 0.06 |
| Source moments, boundary values, full trace | 0.28 | |
| Trace expansion, finiteness check, field wrappers | | 0.14 |
| Wrappers, allocation, logging | 0.10 | 0.08 |

Three changes removed it, and all three are the default code now:

- the device properties are read once per process;
- the launch is chosen once per operator, phase and matrix format;
- the status of the local factorizations is read back only after a kernel
  that factorizes, because stored factors cannot fail.

| Case | Time step before | After | Refresh | Reconstruction |
|---|---:|---:|---|---|
| ARK3, `p = 4`, `N = 1,474` | 74.0 ms | 49.2 ms (-33.6%) | 3.41 to 0.38 ms | 3.27 to 0.24 ms |
| ARK3, `p = 2`, `N = 23,254` | 239.8 | 215.1 (-10.3%) | 3.47 to 0.42 | 3.27 to 0.27 |
| ARK3, `p = 4`, `N = 369,790` | 785.8 | 761.2 (-3.1%) | 5.95 to 2.77 | 5.44 to 2.50 |
| BDF2, `p = 7`, `N = 92,552` | 73.1 | 64.8 (-11.3%) | 8.34 to 5.30 | 8.13 to 5.10 |
| BDF2, `p = 9`, `N = 92,552` | 191.4 | 183.7 (-4.0%) | 17.20 to 14.07 | 16.48 to 13.37 |

What remains after these changes is real work: the source moments, the
boundary values and one synchronization with the device per call.

The ADR assembly had the same cost in another place: choosing its kernel
took 4.4 ms per assembly at `p = 6`, of which 2 ms were two readings of the
device properties. Choosing once per process brought it to 0.04 ms, with
bitwise equal results.

## 2. What limited the kernels at degrees 7 to 9

A replay of the `p = 9` kernels with a clock per stage (`N = 92,552`, stored
LU factors, 128 threads, four blocks per multiprocessor):

| Stage | Refresh | Reconstruction |
|---|---:|---:|
| Whole kernel | 13.6 ms | 13.1 ms |
| Triangular solves, one thread per column | 57% | 59% |
| Coupling product, recomputed for each entry | 16% | 16% |
| Copy of the 24 KB factor to shared memory | 13% | 14% |
| Column build, flux, output | 13% | 11% |

The kernels were limited by waiting within each block, at a quarter of the
possible number of resident threads. They were not limited by memory
bandwidth: reading the factors alone ran at 1.25 to 3.48 TB/s.

The step from degree 8 to degree 9 is a step in shared memory. With eight
blocks of 64 threads, raising the shared memory in use on one
multiprocessor from 176 to 192 KB makes the `p = 8` refresh go from 7.1 to
10.4 ms. The `p = 9` kernel needs 203 KB for eight blocks, so the tuner
chooses four blocks of 128 threads. The likely cause is that the cache
shrinks as shared memory grows; this was inferred, not measured.

## 3. What was tried

Replay at `p = 9`, `N = 92,552`. *Parity* is the largest relative difference
from the unchanged kernel.

| Change | Refresh | Reconstruction | Parity | Outcome |
|---|---:|---:|---|---|
| Unchanged | 13.60 ms | 13.11 ms | | |
| Solve each column on one warp | 7.15 | 6.70 | 2e-12 | **default now** |
| and skip the trace lift of interior elements in the refresh | 4.15 | 6.70 | 2e-12 | **default now** |
| and read the factors in place | 2.55 | 4.33 | 2e-12 | **default now** |
| Stored local inverse, applied by one product | 1.76 | 3.87 | 5e-11 | works, not adopted |
| Contract with the reference tables first | 12.6 to 13.6 | 12.0 to 13.1 | | no gain |
| Load the factor with several reads per thread | 12.6 to 13.6 | 12.0 to 13.1 | | no gain |
| Compiler level `-O1` | 15.7 | 14.8 | | slower |
| At most 80 registers per thread | 10.4 | 10.1 | | small gain, superseded |

In real runs the three adopted kernel changes made a time step 6 to 11%
faster at degrees 7 to 9, and 12 to 19% together with the host changes of
section 1, at parity.

## 4. What to store

Stored LU factors are most of the gain. Removing them from the code of
section 3:

| Case | Time step | Without stored factors | Reconstruction without, with |
|---|---:|---:|---|
| ARK3, `p = 4`, `N = 369,790` | 562 ms | +3.6% | 5.24, 2.80 ms |
| BDF2, `p = 7`, `N = 92,552` | 52.1 | +23% | 9.45, 2.58 |
| BDF2, `p = 9`, `N = 92,552` | 156 | +28% | 28.67, 4.65 |

**Stored responses.** Assembly already solves the local system for every
trace column, and the refresh already solves it for the source. Keeping
these solutions removes the local solve from reconstruction. This is the
option `cache_local_factors="schur-lu+responses"`.

| Case | Time step | Reconstruction | Stored memory |
|---|---|---|---|
| ARK3, `p = 4`, `N = 369,790` | 561 to 553 ms (-1.6%) | 2.79 to 0.85 ms | 0.69 to 1.40 GB |
| BDF2, `p = 7`, `N = 92,552` | 52.2 to 50.7 (-3.0%) | 2.58 to 1.02 | 0.97 to 1.64 |
| BDF2, `p = 9`, `N = 92,552` | 156.0 to 153.0 (-2.0%) | 4.65 to 1.56 | 2.26 to 3.52 |

Setup time and iteration counts do not change, and the fields are at parity.
The refresh is about 3% slower at degrees 8 and 9, because it adds the
response sum.

**Stored inverse.** It only replaces a solve by a product, and the solve is
no longer the largest part of the kernel. It made a time step 0.2 to 2.3%
faster, cost 33 to 69 ms per operator, left a residual of 1e-13 where a
solve gives 1e-15, and was above the parity floor on one case (2.3e-10
against 6.2e-11). Not adopted; kept as a switch that is off.

**Cholesky factors.** The Cholesky option reconstructs in 1.2 ms at `p = 9`
and, with a warp solve, refreshes in 2.4 ms. But its potential differs from
the LU result by 5 to 6e-10 relative, and the error of the gradient of a
manufactured potential changes (2.25e-7 to 4.13e-7 at `p = 7`). The stored
responses reproduce a full Cholesky solve to 5e-15, so the difference is in
the factorization itself. Which factorization is closer to the exact local
solution was not established. The LU option stays the reference.

## 5. Where the code is

| Result | Where |
|---|---|
| Device properties read once | `device_signature` in `hybridge/runtime/cuda_tuning.py` |
| Launch chosen once per operator | `_launch` in `hybridge/mixed/raw_cuda/poisson.py` |
| Warp solve, factors in place, interior skip | `hybridge/mixed/raw_cuda/poisson_source.py` |
| Stored responses | `cache_local_factors="schur-lu+responses"` |
| Everything not adopted | switches that are off by default, listed at the top of `hybridge/mixed/raw_cuda/poisson.py` |

The launcher of the study is
[`campaigns/poisson_element_cost`](../../campaigns/poisson_element_cost/README.md).

## 6. Limits

- One GPU model, shared with other jobs during the measurements. All
  comparisons are paired within one job, and the unchanged code repeated
  stays within 0.8%.
- The shared-memory step is inferred from a padding test.
- Not measured on compute capabilities 7.5 and 12.0.
- Stored responses were measured for constant diffusion with source and
  boundary updates only. Reconstruction must follow a refresh of the same
  source.
