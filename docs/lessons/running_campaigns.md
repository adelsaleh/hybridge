# Lessons from running campaigns

A campaign is a set of runs started by one launcher under `campaigns/`, on a
batch system, often for hours. These lessons are about keeping such runs
correct, restartable and cheap. They are written for any cluster; where a
number is given, it was measured once and shows the size of the effect.

## Before the first run

1. **Time the start of a worker, not only its work.** A worker that did one
   second of work took 153 s to start. Over half of that was a library
   search through a Python environment on a network file system, and the
   rest was compiling about 1,100 modules that had no bytecode. Giving the
   library path explicitly, precompiling into a local cache and importing
   only what the worker needs brought the start to under half a second.
2. **A copied source tree needs its root on the import path.** A frozen copy
   of the repository imported `campaigns...` only because one helper script
   set `PYTHONPATH`. Set it in the launcher.
3. **Compile-only runs need a loose tolerance.** A run whose purpose is to
   fill the kernel caches should pass `rtol = 1`. Otherwise a toy mesh that
   misses its residual stops the run before the later kernels compile.
4. **Mesh generation needs its own tests at the target size.** A floor on
   the element size, harmless on small meshes, capped the largest mesh at
   1.75 million triangles instead of the size requested.

## Memory and threads

5. **Do not count reclaimable file cache as used memory.** A memory limit
   computed from the job's current usage excluded a case that needed 90 GiB,
   because 148 GiB of file cache counted as used. Subtract the file cache or
   give the limit explicitly.
6. **Keep threaded BLAS out of threaded loops.** Calling a multithreaded
   BLAS from inside a parallel Numba loop crashed at degree 9. Set the BLAS
   to one thread there. One BLAS build also has a fixed number of per-thread
   buffers and aborted above about 128 calling threads.
7. **Host assembly scales when its kernels call no BLAS.** One assembly took
   102 s on 16 threads and 17.4 s on 192. The device route was still 4 to 7
   times faster.
8. **Thread tuning is noise on small systems.** Below about 100,000 DOFs the
   direct-solver times were flat within 10% from 8 to 96 threads. Tune on a
   few large systems.

## File systems

9. **Writing large arrays to a shared file system can stall for minutes.**
   Exports of result arrays paused about five minutes per file. Writing
   them in a background thread, joined after the last solve, kept the GPU
   busy meanwhile.
10. **Many small new files can exhaust a quota that has room.** On some
    network file systems every file being written reserves a large block
    until minutes after the last write. A test suite that wrote a thousand
    small files, 12 MiB in all, raised the quota reading by 156 GiB within a
    minute. Put temporary directories and compile caches of test suites on a
    scratch file system, and bundle records into one archive per campaign.
11. **Lock the output directory, not only the device.** Two jobs on
    different GPUs wrote into the same stage directory, because the only
    lock was per GPU.

## Restarting and stopping

12. **Completion markers make a campaign restartable.** Each finished run
    writes a marker, and the same command skips the runs that have one. A
    stage that was replaced by another needs a marker too, or it is retried.
13. **A check that hashes absolute paths fails in another checkout.** Meshes
    built from one copy of the repository looked "changed" from another.
    Hash contents and relative paths.
14. **Cancelling a job does not stop a chain.** A job that is the successor
    of the cancelled one, or one that resubmits itself, continues the
    campaign. Cancel or hold those as well.
15. **A non-zero exit status is not a crash.** A stage that exits with 1
    whenever one of its runs did not pass has still finished. Read the
    counts.
16. **A gate only blocks if the next stage checks it.**

## Sharing GPUs

17. **A job that waits for work still holds its GPU.** A stage runner with an
    idle limit of twelve hours kept a GPU that nothing used. Twenty to
    thirty minutes is enough.
18. **Count GPUs over everything that is running or queued,** including
    short tests submitted by hand.
19. **Split work by the unit that is compared.** When each GPU runs whole
    rows of a comparison, no ratio mixes two devices, and five GPUs cut a
    campaign from 4.3 hours to 54 minutes.

## Planning the size

20. **Cost grows faster than the mesh.** A campaign that finished 55 runs
    per hour on small meshes finished 7 per hour at about 100,000 triangles.
    Estimate the largest level first and run the largest systems early.
