# Persistent CUDA autotuning for the face-dense solver

The T600 measurements show that kernel fusion is architecture- and
problem-dependent.  `raw_fused` reduces the operator workspace but does not
always beat `raw`; the same is true for the fused additive-Schwarz path at
higher trace order.  The production solver therefore must select these paths
by measurement rather than by a fixed global default.

## Interleaved candidate timing

`autotune_face_dense_gpu` now constructs every numerically equivalent
candidate first, verifies its result against the reference candidate, and then
records CUDA-event samples in alternating order:

```text
repeat 0: raw -> fused
repeat 1: fused -> raw
repeat 2: raw -> fused
...
```

This reduces bias from GPU clock changes, thermal state, and always timing one
implementation first.  Setup and factorization remain outside the timed
region; the autotuner selects the repeated application path used inside GMRES.

## Persistent cache key

A cached decision is valid only when all of these fields match:

- kernel ABI version;
- GPU name and compute capability;
- total device memory;
- CUDA driver and runtime versions;
- floating-point dtype;
- number of system rows and neighbour slots;
- face block size;
- boundary mode;
- polynomial order;
- local factorization strategy;
- operator candidates;
- ASM candidates.

The exact key is hashed with SHA-256.  A T600 entry can therefore never be
silently reused on a P100 or V100, and a driver/runtime change creates a new
entry automatically.

The default file is:

```text
~/.cache/hdgfem/face_dense_autotune.json
```

It can be overridden with the `HDGFEM_AUTOTUNE_CACHE` environment variable or
with `--cache-file` in the command-line script.  Writes use a temporary file
and `os.replace`, so an interrupted write cannot leave a partially written
cache file.

## Command-line use

First run on a new device/problem configuration:

```bash
PYTHONPATH=. python scripts/autotune_face_dense_gpu.py \
    --mesh 64 \
    --order 4 \
    --boundary-mode eliminate \
    --warmup 20 \
    --repeats 100 \
    --cache-file results/gpu_autotune_cache.json \
    --output results/t600_autotune_64_p4.json
```

The first invocation reports `cache: miss` and stores the selected operator and
ASM path.  Repeating the same command reports `cache: hit` and skips the CUDA
microbenchmarks.

Force a new measurement after a code or experimental change:

```bash
... --force-retune
```

Disable persistence while debugging:

```bash
... --no-cache
```

On the mesocentre, use a separate cache file for each campaign directory.  The
device fingerprint still prevents accidental cross-architecture reuse.

## Production GMRES fallback correction

CGS-to-CGS2 fallback is now activated only when another restart cycle will be
executed.  If the true residual already satisfies the tolerance at the end of
the current cycle, the solver returns as converged without counting a fallback
that can never be used.  This removes the misleading `fallback_count = 1`
previously printed for one-cycle degree-18 polynomial solves.

## Local-assembly order guard

The local-assembly validation script now checks the requested polynomial order,
element basis size, and trace basis size before performing any GPU work.  It
also prints both basis dimensions.  A run requested with `--order 4` must print:

```text
Mesh / order        : 64x64 / p=4
Element / trace dofs: 15 / 5
```

A log that prints `p=1` and `3 / 2` did not validate the order-four path and
must not be used as an order-four result.
