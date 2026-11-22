# Plan: kernel choices measured once and shipped with the package

## Goal

Someone who installs HYBRIDGE on a GPU of a known architecture should get the
fastest known kernels in the first run, with no measuring delay and no
settings to look up. Someone on an unknown architecture should get a safe
start, a measurement on first use as today, and one command that turns their
measurements into a file that can be added to the package for the next user.

## What exists today

The tuner, `hybridge/runtime/cuda_tuning.py`, already measures and stores.
On first use of a workload it times at most eight candidates within 30
seconds on a sample of the real data, checks each against a reference
result, and writes one JSON record to `~/.cache/hybridge/cuda-tuning`. A
record holds:

- `device`: name, UUID, compute capability (`major`, `minor`), number of
  multiprocessors, shared memory per block, registers per block, the
  FP32 to FP64 ratio, and the driver, runtime and CuPy versions;
- `workload`: kernel family and phase, degree, bases, matrix format, cache
  option, precision, the mesh-size class and the hash of the kernel source;
- the candidates, their timings, and the chosen one.

Two things keep this knowledge on one machine:

1. **The key of a record contains the device UUID and the driver versions.**
   A record is never reused on another GPU, even of the same model.
2. **What was learned across machines is written in code.** The candidate
   order for GPUs with full-rate FP64 is the function
   `_fp64_rich_assembly_priors` in `hybridge/mixed/raw_cuda/poisson.py`; the
   thread grids of the register solve are the table `DEFAULT_CONFIGS` in
   `hybridge/linalg/gpu/fused_batched_solve.py`; the choice of the tensor
   route is a test on compute capability 9.0 in `resolve_raw_local_assembly`
   (`hybridge/solvers/capabilities.py`); the wide ADR launch is
   `wide_variable_launch` in `hybridge/mixed/raw_cuda/tensor.py`. A new
   architecture means editing these by hand.

## Design

### 1. A shipped choice

A shipped choice is a tuner record with the machine removed. It keeps the
workload, the chosen candidate, the candidate timings, the kernel source
hash and the HYBRIDGE version, and replaces the device by a *device class*:

| Field of the class | Why it is in the key |
|---|---|
| compute capability (`major.minor`) | the same source compiles to different register counts |
| FP32 to FP64 ratio | decides whether dense local algebra is limited by arithmetic |
| shared memory one block may request | decides which memory layouts fit |
| number of multiprocessors, as a range | decides how many blocks run at once |

The device name is kept as information and is not part of the key. The UUID
and the driver versions are dropped.

The files are package data, one per class, under `hybridge/data/tuning/`,
and are installed with the wheel.

### 2. How the tuner uses it

For a workload with no record in the machine-local cache, the tuner looks
for a shipped choice of the same workload, first for the same class, then
for the same compute capability, then for the same FP32 to FP64 range.

- With `HYBRIDGE_CUDA_TUNING=auto`, the shipped choice is the first
  candidate after the conservative baseline. It is timed and checked like
  any other, and the result is stored in the local cache as today.
- A new mode, `HYBRIDGE_CUDA_TUNING=shipped`, checks the shipped choice
  against the reference result once and uses it without timing the others.
  This is the mode with no first-use delay. It falls back to `auto` when no
  shipped choice exists.
- `off` keeps its meaning: the first candidate, no trial.

A shipped choice is never trusted for correctness. A candidate that fails
its check is rejected whatever its origin.

A shipped choice whose kernel source hash differs from the installed source
was measured on other code. It is then used as the first candidate under
`auto`, and ignored under `shipped`.

### 3. One command to produce the file

`python -m hybridge.runtime.tuning_export OUTPUT.json` runs a fixed list of
workloads on the current GPU with the exhaustive candidate list: the three
equation families, their phases, degrees 1 to 9, the matrix formats and the
cache options. It writes one file in the shipped format. The same command
with `--from-cache` converts the records already in the local cache.

Adding that file under `hybridge/data/tuning/` is the whole contribution. A
test checks its fields, and that every chosen candidate still exists in the
code.

### 4. One source for code and documentation

The hand-written tables listed under "What exists today" are replaced by
lookups in the shipped files. The settings tables of
[kernel tuning by GPU architecture](../../lessons/kernel_tuning_by_architecture.md)
are then produced from the same files by a script under `scripts/reports/`,
so that the page cannot disagree with what the package does.

## Steps

1. Define the shipped format and the device class; write the reader and the
   lookup, with tests on recorded files. No behaviour changes yet.
2. Add `--from-cache` export and convert the records of the capability 9.0
   GPU. Check that `auto` with these files reproduces today's choices.
3. Add the `shipped` mode. Acceptance: on that GPU, the first solve of each
   standard workload starts no timing trial and its kernel time is within 3%
   of the tuned one.
4. Replace the four hand-written tables by lookups, one at a time, each with
   a comparison of the chosen candidates before and after.
5. Add the full export command and the test of contributed files.
6. Generate the documentation tables.
7. Export on compute capabilities 7.5 and 12.0, where the present defaults
   were never qualified.

## Open questions

- **Two GPU models in one class.** Two models of the same compute capability
  can differ in memory bandwidth. If their choices differ, the class needs
  another field, or the file must hold both.
- **Mesh size.** A choice is tied to a mesh-size class. Whether the choice
  for one class is a good first candidate for its neighbours has to be
  measured.
- **Kernel changes.** Every edit of a kernel changes its source hash and
  turns the shipped choices for it into hints. The export of the reference
  architectures has to be part of changing a kernel.
- **File size.** The full list is a few thousand workloads per class. The
  timings may have to be dropped from the shipped file and kept with the
  campaign records.

## Not in scope

Shipped choices select among kernels and launches that give the same result.
They never change precision, quadrature, a solver or a tolerance.
