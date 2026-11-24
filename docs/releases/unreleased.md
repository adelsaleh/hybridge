# Changes after 0.1.0a2

This page lists, one change at a time, everything on `master` after release
`0.1.0a2` and after the guiding-center research code moved to its own
repository (`f371e95`). The package version is still `0.1.0a2`: none of this
is on PyPI yet.

Each entry says what changed, how to select it and where it is documented.
The settings measured for each GPU architecture, and
what the measurements taught, are collected in
[kernel tuning by GPU architecture](../lessons/kernel_tuning_by_architecture.md).

Three words are used throughout. A *launch* is the number of threads per
block with which a CUDA kernel is started. The *tuner* is the code in
`hybridge/runtime/cuda_tuning.py` that times a few equivalent kernels or
launches on a sample of the real data and keeps the fastest one. An *opt-in*
feature does nothing unless an option or an environment variable asks for it.

## Assembly on the GPU

1. **Advection-reaction: tensor assembly.** `raw_local_assembly="tensor"`
   builds the local matrices of all elements as batched tensor contractions
   (cuTENSOR, or cuBLAS matrix products) and solves them in batches.
   `"auto"` chooses it on GPUs of compute capability 9.0 from degree 5, when
   the advection field is a DG field in a compatible space and the work
   arrays fit in a quarter of the free device memory. The elements are
   processed in chunks whose work arrays take at most 16 GiB
   (`HYBRIDGE_RAW_TENSOR_CHUNK_GIB`), so the memory no longer grows with the
   mesh. See [assembly](../algorithms/assembly/README.md), section 2.3.
2. **Advection-reaction: batched local solve in registers.**
   `hybridge/linalg/gpu/fused_batched_solve.py` solves
   `A_K R_K = [B_K | f_K]` for every element `K` by Gauss–Jordan elimination
   with partial pivoting. One thread block holds one augmented matrix in
   registers. It is FP64 only. The tensor assembly times it against cuBLAS,
   MAGMA and the cooperative LU and keeps the fastest
   (`HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER=auto`, the default).
3. **Advection-reaction: volume term of the three-kernel assembly as one
   matrix product.** With `HYBRIDGE_RAW_SPLIT3_VOLUME=dmma` or `cublas`, the
   dense volume operator of `raw_local_assembly="split3"` at degrees 8 and 9
   is computed for all elements by one matrix product. The default,
   `inline`, is unchanged.
4. **Advection-diffusion-reaction: explicit inverse of the variable
   full-tensor mass matrix.** From degree 7 the kernel inverts this
   `2 NEL x 2 NEL` matrix once per element and applies the inverse by
   products, where it used to factor it and solve repeatedly.
   `HYBRIDGE_RAW_ADR_MASS_INVERSE=off` restores the LU factorization.
5. **Advection-diffusion-reaction: launch and memory layout.** On compute
   capability 9.0 the automatic launch for variable tensors uses 256 threads
   at degree 8 and 512 at degree 9. `HYBRIDGE_RAW_ADR_TENSOR_MASS` chooses
   whether the mass factor lives in shared or in global memory, and
   `HYBRIDGE_RAW_ADR_PHASE_CLOCK=1` builds a diagnostic kernel that reports
   the cycles spent in each of its 16 stages.
6. **Advection-diffusion-reaction: the kernel is chosen once per process.**
   The choice of kernel, layout and launch for one device, space, trace
   space, diffusion classes, matrix format and cache option is made at the
   first assembly and kept (`resolved_selection`). Before, each assembly
   repeated the lookup, which took about 4 ms. See
   [Poisson local solves](../lessons/poisson_local_solves.md), section 1.
7. **Poisson: candidate order for GPUs with full-rate FP64.** When FP32 is at
   most 4 times faster than FP64, the tuner first times the launches that
   won on such a GPU: all right-hand-side columns in one batch, and a thread
   count that grows with the element size. Other GPUs keep the earlier
   order.
8. **Poisson with stored LU factors: faster right-hand side and
   reconstruction.** These two kernels now solve each local column on one
   warp and read the stored factors in place. The right-hand-side kernel
   skips the trace lift of elements without a boundary edge, because the
   trace it is given holds boundary values only. The launch is chosen once
   per operator, and the device properties are read once per process.
   See [Poisson local solves](../lessons/poisson_local_solves.md).
9. **Poisson: `cache_local_factors="schur-lu+responses"`.** This option
   keeps, besides the LU factors, the local solutions that assembly already
   computes for every trace column. Reconstruction then needs no local
   solve. It takes more device memory than `"schur-lu"`.
10. **Poisson: experimental switches.** The parts of the element-cost study
    that were declined or that failed are kept as environment switches that
    are off by default: a stored local inverse, a warp solve for the
    Cholesky option, two reordered products, a wider factor load, extra
    compiler options and three diagnostics. With none set, the kernels are
    token for token the default ones. The table of every item and its result
    is in the [campaign README](../../campaigns/poisson_element_cost/README.md).
11. **Vendored CuPy ADR assembler: device route.** The assembler under
    `vendor/adr_gmres` now samples coefficients, builds local matrices,
    finds the face topology and eliminates Dirichlet data on the GPU, in
    element chunks of at most 2 GiB (`HDGFEM_CUPY_ADR_WORKSPACE_GIB`).
    `HDGFEM_CUPY_ADR_ROUTE=host` keeps the earlier route.
12. **Host ADR preparation on all cores.** The host preparation of the ADR
    assembly and the total-flux postprocess run in multithreaded Numba
    (`mixed/coefficients_numba.py`, `mixed/postprocess/total_flux_numba.py`),
    and the coefficients of the catalogue cases are compiled
    (`cases/manufactured_pointwise.py`, `cases/closed_loop_pointwise.py`).
13. **Sources given at quadrature points.** Every diffusion and transport
    backend accepts a `QuadratureScalar` as the source, that is, values at
    the solver's own quadrature points, with no projection onto the
    polynomial space.

## Linear solvers

14. **hp-multigrid: kernel presets.** `kernels=` on `FaceBlockHpMgPcgSolver`
    and on `BernsteinHpSymmetricPartPreconditioner`, or
    `HYBRIDGE_HP_MG_KERNELS`, selects `legacy` (default), `packed` (one
    thread per scalar row, same arithmetic as `legacy`), `packed-fused`
    (residual and restriction in one kernel) or `graph` (each sweep replayed
    as one CUDA graph).
15. **Face-dense operator, Schwarz and block-Jacobi kernels: two more memory
    layouts.** `implementation="coalesced"` or `"interleaved"` on the
    operator, and `application=` with the same values on the
    preconditioners, read the matrices in the order in which they are
    stored. `"interleaved_f32"` stores the preconditioner inverses in FP32
    and sums in FP64.
16. **GMRES: stop at the requested residual after a warm start.** With
    `estimate_target="true_target"` the inner stopping test is scaled to the
    target `rtol * ||b||`. The default test is relative to the residual at
    the start of each restart cycle, so a solve that starts from a good
    guess iterates down to round-off.
17. **AMGX: matrices are borrowed from device memory.** With the AMGX and
    pyamgx forks, `PyAMGXCsrDeviceSolver` hands a device CSR matrix, or a BSR
    matrix with square blocks, to AMGX without copying it. A matrix the fork
    cannot borrow is uploaded as before. Plain `JACOBI_L1` is refused on
    block rows, because AMGX needs scalar rows for it.
18. **AMGX: stop relative to `||b||`.** A configuration whose convergence
    test is `RELATIVE_RHS` stops when the residual falls below
    `rtol * ||b||`, so a good initial guess saves iterations. The kernel that
    scales the rows of a BSR matrix is now timed per block size.
19. **AMGX: coarsest level of the Chebyshev Poisson configuration.**
    `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json` sets
    `dense_lu_num_rows` to 256 instead of 2048, so the level that AMGX solves
    by dense LU in every cycle is smaller.
20. **ADR: further AMGX attempts after a rejected solve.**
    `amgx_retry_attempts` on the ADR options, with raw-CUDA assembly only,
    lists the attempts to make when a solve is rejected.
21. **Replaying a saved failed system.**
    `hybridge.linalg.failure_snapshot.load_system_snapshot(path, device=...)`
    reads the matrix, right-hand side and saved iterates of an archive written
    after a failed solve into a `SavedReducedSystem` (other arrays, such as the
    mesh and string metadata of transport archives, stay in the file); it can be
    passed back to `solve_reduced_system_amgx_device`. Its
    `relative_residual(x)` computes ||b - A x||_2 / ||b||_2 on the host with SciPy,
    independently of the solver that produced x. With
    `record_attempt_details=True`, `solve_reduced_system_amgx_device` adds to
    every attempt record its complete AMGX residual history and a host copy of
    its iterate; it is off by default because step records store the attempts.
22. **Host PARDISO as the last device retry.**
    A retry `{"backend": "pypardiso"}`, placed last like `cusolver-qr`, copies
    the original unscaled system to the host as CSR, solves it with oneMKL
    PARDISO (general LU, `MKL_NUM_THREADS` threads) and accepts the solution only
    if ||A x - b|| measured again on the device meets the physical tolerance.

## Kernel tuning

21. **One tuner for every kernel family.** `hybridge/runtime/cuda_tuning.py`
    times at most eight candidates within 30 seconds on first use
    (`HYBRIDGE_CUDA_TUNING_CANDIDATES`, `HYBRIDGE_CUDA_TUNING_SECONDS`),
    checks each result against a reference, and stores the choice in
    `~/.cache/hybridge/cuda-tuning`. It serves the Poisson kernels, the
    tensor local solves, the three assembly routes of advection-reaction,
    the ADR mass layout and the solver kernels. See
    [architecture tuning](../development/architecture_tuning.md).
22. **Measured launches for three postprocessing kernels.** The Poisson
    primal recovery, Oswald averaging and the continuous gradient time the
    block sizes 32 to 512 once per process and workload (`launch_tuned`).
23. **Compiled kernel resources are recorded.** The registers and shared
    memory of a compiled kernel are read back, and the tuner keeps them with
    its records.

## Postprocessing

24. **ADR recovery operators (opt-in).**
    `hybridge/mixed/postprocess/adr_recovery.py` provides the reusable
    pieces of the ADR primal recovery and total-flux correction: an assembly
    by matrix products against stored reference data, and a correction map
    that is set up once per mesh.

## Runtime and display

25. **Log rows with times.** The log prints one row per startup step with an
    aligned time column, and from verbosity 3 it says when a CUDA kernel
    first runs and whether its launch was measured, read from the cache,
    given explicitly, or left at its fixed value because tuning is off.
26. **Holoviz without a display on the GPU.** With `frame_window=True` the
    frames rendered without a window are shown in a small Tk window over
    X11. The window keeps only the newest frame, so a slow connection does
    not delay the solver. See [Holoviz](../backends/holoviz.md).

## Campaigns, lessons and repository layout

27. **Campaign layout.** Launchers, job scripts and READMEs are in
    `campaigns/<name>/`, and the shared launcher code in `campaigns/common/`.
    A launcher names no machine. It takes the repository from
    `HYBRIDGE_REPO` or from the directory it is submitted from, and writes
    bulk output under `HYBRIDGE_SCRATCH`, or to `run_outputs/` without it.
28. **Campaign records stay on the machine that ran the campaign.** What a
    launcher writes to `docs/campaigns/<name>/` (job records, logs, result
    tables and notes) names nodes, jobs and local paths, and git ignores it.
    The records of three studies that were already published (`adr_pardiso`,
    `adr_stress` and `adr_unified`) stay tracked; they moved from
    `campaigns/<name>/records/` to `docs/campaigns/<name>/records/`.
29. **Lessons.** [`docs/lessons/`](../lessons/README.md) holds what the
    campaigns taught, written for any machine: the kernel settings per GPU
    architecture, the cost of the Poisson local solves, the solver studies,
    and running long batch campaigns.
30. **New campaigns.** `adr_h200_tuned` (iterative ADR solves on a ladder of
    meshes), `adr_pardiso` and the host LU references, `cpu_adr_scaling`
    (thread scaling of the host assembly), `solver_h200` (solver kernels),
    `architecture_tuning` (qualification of the tuner), `fork_integration`
    (qualification of the AMGX and pyamgx forks), `matk`,
    `poisson_element_cost`, `adr_selection_cost` and `adr_recovery`. Each
    has a README with its commands.
31. **Reports.** The ADR solver studies and their figures are under
    `docs/research/solver_studies/`, and the assembly studies per GPU
    architecture under `docs/algorithms/assembly/`.
32. **Generated builds.** Build trees are grouped in a `builds/` folder
    beside the repository, one folder per project and variant.
33. **Generic paths.** Documents and scripts name files relative to the
    repository root or to its parent directory. Where an absolute path is
    needed they use `/home/user` and `/scratch/user`, and compute nodes are
    called `gpu-node` and `cpu-node`.
34. **Plan: kernel choices shipped with the package.** The measured choices
    per architecture are to become package data that the tuner reads, with
    one command to produce the file on a new GPU. See the
    [plan](../development/plans/shipped_kernel_choices.md).
35. **The n-Gamma model left the package.** The n-Gamma D-BDF2 model, its
    runner and its tests (`scripts/n_gamma/`, `tests/test_n_gamma_*.py`)
    moved to a separate repository, `ngamma`, that uses Hybridge as a
    library. The `manufactured` extra in `pyproject.toml`, which only that
    model needed, is gone.
36. **Job scripts name no Slurm account.** The `#SBATCH --account` lines are
    gone from the job scripts, and the submit scripts pass `--account` only
    when `CAMPAIGN_ACCOUNT` is set. Without it, Slurm uses `SBATCH_ACCOUNT` or
    the user's default account. The partitions (`hopper`, `genoa`) remain the
    defaults and are changed with `CAMPAIGN_PARTITION` or `sbatch --partition`.
37. **Campaign pages without run history.** The campaign READMEs give the
    purpose, the commands, the options and the outputs. Job numbers, run
    status, the names of local copies and the ids of unpublished commits stay
    with the local records of the machine that ran the campaign.

## Fixed

Bugs found while testing this change. Each entry gives what was
seen, the cause and the fix.

38. **Illegal memory access in the ADR tensor tuner.** Seen: tests that run
    the tuner on a matrix that is singular on purpose sometimes stopped with
    `cudaErrorIllegalAddress`, and every later CUDA call in the same process
    then failed. Cause: when the tuner tries the variant that stores the
    local LU factors, an element whose factorization fails leaves the kernel
    before it writes its pivots. The trial reconstruction still reads those
    pivots, and uses them as row indices in shared memory
    (`lu_solve_setup`). The pivot buffer was allocated without
    initialization (`cp.empty`), so the indices were whatever the memory
    pool held before. Fix (`hybridge/mixed/raw_cuda/tensor_tuning.py`): the
    trial pivot buffer starts at zero, which is always a valid row. A failed
    factorization in the sample now raises `UnrepresentativeInput`
    (`hybridge/runtime/cuda_tuning.py`): the tuner then returns its default
    without timing the other candidates, and keeps nothing in its caches,
    so the next call on valid data is measured. Before, the sample failure
    counted against the candidate and the whole selection failed. Test:
    `tests/test_cuda_tuning.py`.
39. **`release_device` raised on a GPU in a bad state.** Seen: a campaign's
    end-of-task GPU check stopped with `ValueError: invalid literal for
    int() with base 10: '[N/A]'`. Cause: `nvidia-smi` prints `[N/A]` for a
    field it cannot read, and a faulty GPU prints it even for process ids;
    `campaigns/common/lifecycle.py` converted every process id with `int()`.
    Fix: unreadable numbers are kept as `None`. When a process id cannot be
    read, the process cannot be assigned to a job, so `release_device`
    records `released=None` with the reason instead of raising. The same
    change protects the memory heartbeat of `campaigns/h100_assembly/run.py`.
    Test: `tests/test_campaign_lifecycle.py`.
40. **Tests that no longer matched the code.** Four device tests checked
    behaviour that had changed on purpose: the H200 node-LU report now
    computes its FP64 floor from `l2_error` and the LU status
    (`tests/test_adr_h200_node_lu.py`); quadrature-point coefficients are
    rejected with a message that names the routes that accept them
    (`tests/test_h200_expansion.py`); the shared memory of the
    over-integrated tensor kernel is checked against the formula the kernel
    uses (`tests/test_adr_tensor_raw_cuda.py`); and `"auto"` local assembly
    times the equivalent routes instead of following a degree threshold
    (`tests/test_transport_cross_space_advection.py`). Functions without
    docstrings in `io/`, `runtime/` and `mixed/postprocess/` received them.
41. **Broken links in the published `adr_unified` reports.** Seen: 24 links
    in the four READMEs under
    `docs/campaigns/adr_unified/records/published_reports/` pointed to missing
    files. Cause: the records moved one folder deeper (entry 28), and their
    relative links kept the old depth. Fix: each link has one more `../`, and
    every target was checked to be a tracked file.

Commit ids quoted in other documents refer to the development history, which
is kept locally and is not published.

## Known gaps

- On a machine without a CUDA driver, the device tests of several files stop
  with an error where they should be skipped.
- The tuner's choices were qualified on one GPU of compute capability 9.0.
  The same qualification on compute capabilities 7.5 and 12.0 is pending.
