# GC article convergence campaign

This campaign reruns every GPU convergence run of the guiding-center article (Section 5 and Appendices B and C).
Each run records the accurate error rule and saves its final fields. The spaces' own volume rule integrates only
degree 2p exactly, so it underestimates `||rho_h - rho||`. The runs record the errors of that rule and, in the
`*_accurate` columns, the errors with 2p + 2 collapsed-Gauss points per direction. Every run also saves its final
state (`fields/<run>_final.npz`), so later error evaluations need no rerun.

The campaign is prepared on a workstation and meant for an H100 cluster (Jean Zay, SLURM). It can also run locally.

## What runs

The default case is the k = 8π travelling wave: IMEX-ARK3, T = 1/16, Poisson τ = 1, conflict-averaged upwind,
device profile `bsr` (raw-CUDA assembly, AMGX at rtol 1e-11), and the article time steps (`dt_table.json`).
The exact drift is the pointwise one: the transport receives the exact drift at its volume and face quadrature
points (`hybridge.core.quadrature_coefficients.QuadratureVectorField`), with a single-valued normal flux, through the
same raw-CUDA kernels. The Poisson problem is still solved; only the drift changes. Its DG(p + 2) projection
(`--exact-drift-mode projected`) is the cross-check of group `exact-drift-projection-check`.

| Group | Runs | Local hours | Purpose |
|---|---:|---:|---|
| `smoke` | 12 | 0.6 | every code path, a few steps on coarse meshes; run first on a new machine |
| `lattice` | 2 | 0.2 | edge-lattice orientation of the Gmsh meshes (no simulation, no GPU) |
| `gmsh-raw` | 26 | 3.3 | Gmsh ladders p = 2–6, raw drift q = p |
| `gmsh-raised` | 52 | 8.4 | Gmsh ladders, raised Poisson degree q = p + 1 and p + 2 |
| `gmsh-compatible` | 26 | 3.0 | Gmsh ladders, compatible drift V2 (Oswald average of the p + 1 postprocessed potential) |
| `gmsh-compatible-primal` | 26 | 3.4 | Gmsh ladders, Oswald average of φ_h itself |
| `gmsh-exact` | 26 | 3.3 | Gmsh ladders, pointwise exact drift |
| `gmsh-finest` | 5 | 1.1 | p = 2 on 2,950,142 triangles: raw, q = p + 1, compatible, compatible-primal, exact |
| `structured-raw` | 32 | 7.1 | aligned and misaligned ladders, raw q = p; p = 3 extended to h = 0.0025, p = 4 to h = 0.005 |
| `structured-raised` | 60 | 9.3 | q = p + 1 on the extended ladders, q = p + 2 on the article ladders |
| `structured-exact` | 32 | 7.1 | extended ladders, pointwise exact drift |
| `structured-compatible` | 16 | 3.4 | misaligned extended ladders, compatible drift V2 |
| `structured-compatible-primal` | 16 | 3.1 | misaligned extended ladders, Oswald average of φ_h |
| `unstructured-raw` | 34 | 4.4 | Gmsh Delaunay ladders p = 2–4, q = p and q = p + 1 |
| `unstructured-compatible` | 17 | 2.1 | Gmsh Delaunay ladders, compatible drift V2 |
| `unstructured-compatible-primal` | 17 | 2.3 | Gmsh Delaunay ladders, Oswald average of φ_h |
| `unstructured-exact` | 17 | 2.0 | Gmsh Delaunay ladders, pointwise exact drift |
| `exact-drift-projection-check` | 16 | 2.4 | projected DG(p + 2) exact drift on the two finest meshes of each Gmsh and misaligned ladder |
| `exact-bdf3` | 8 | 0.2 | BDF3 transport with the exact drift and no Poisson solve (the earlier q = ∞ rows) |
| `temporal` | 2 | 4.5 | temporal studies: the wave (p = 6, Gmsh h = 0.0125) and the orbiting vortex (p = 6, cross polygon h = 0.00625, exact errors) |
| `diagnostics` | 4 | 1.1 | error floor at AMGX rtol 1e-13 (Gmsh p5 q5 h0.01, p4 q5 h0.005); boundary-layer snapshots of the aligned mesh (p = 3, q = 3 and 4) |

That makes 116 tasks and 446 runs, about 72 GPU-hours on the local RTX PRO 5000. The H100 time will differ.
`campaign.py list` prints every task. Every ladder task adds a dt/2 run on its finest mesh, except those of
`smoke`, `gmsh-finest`, the projection check and the diagnostics.

Hours come from `cost_estimates.json`. `campaign.py costs` builds that file from the wall times of the local
runs: it scales the recorded run on the same mesh by the step count, or else uses a per-triangle-step rate of the
two largest recorded meshes. The estimates only balance the shards.

Peak device memory, measured locally: p = 2 with q = 3 on 1.69M triangles took 32.9 GB, which scales to about
57 GB on the 2.95M-triangle mesh. p = 3 with q = 4 on 420k triangles took 14.2 GB, about 43 GB on 1.28M
triangles. One 80 GB H100 holds every run.

## Files

- `campaign.py`: task registry and commands (`list`, `plan`, `status`, `run`, `combine`, `verify-meshes`, `deps`,
  `costs`). One task is one independent ladder: one invocation of
  `scripts/guiding_center/studies/convergence/run_poisson_degree_study.py` (one worker process per run), or one temporal,
  BDF3 or snapshot run. A task is done when every per-run output it expects exists. The driver exits 0 even when
  a run fails, so the exit status alone does not decide.
- `dt_table.json`: the article time steps, plus p = 3 at h = 0.0025 (the article rule, extrapolated log-log) and
  p = 2 at h = 0.00177 (the showcase run). `dt_table_exact_bdf3.json`: time steps of the BDF3 cross-check.
- `meshes.py`, `mesh_manifest.json`: the 16 cached Gmsh meshes (239 MB). The manifest gives family, h,
  algorithm, triangle count, cache file, size and SHA-256 of each. Structured meshes are built
  deterministically and are not cached.
- `cost_estimates.json`: estimated hours per task.
- `analyze_boundary_layer.py`: the article's boundary-layer analysis, run by `combine` (no simulation).
- `slurm/job.slurm`, `slurm/submit.sh`, `env.sh.example`: SLURM array submission, one shard per H100.

## Requirements

- A commit of the repository with every file of `campaign.py deps`; nothing else from the working tree is needed.
- The AMGX and pyamgx forks of the three-repository stack, built for `sm_90` by the user (see
  `docs/getting_started/forked_amgx_stack.md`), CuPy for CUDA 13, Numba, and the package installed in the
  environment of `env.sh`. Gmsh is needed only if a mesh is missing from the cache.
- The mesh bundle (below), unpacked into the repository root.

## Mesh bundle

On the workstation (the manifest is committed; regenerate it only if the meshes change):

```bash
python campaigns/gc_article_convergence/meshes.py verify
python campaigns/gc_article_convergence/meshes.py pack gc_article_meshes.tar.gz
```

On the cluster, in the repository root:

```bash
tar -xzf gc_article_meshes.tar.gz      # writes .cache/hybridge/meshes/*.npz
python campaigns/gc_article_convergence/campaign.py verify-meshes
```

`campaign.py run` checks the meshes of each task before starting it. A task whose meshes are missing or
differ is marked failed without running.

## Running on the cluster

```bash
cp campaigns/gc_article_convergence/env.sh.example campaigns/gc_article_convergence/env.sh   # then edit it
python campaigns/gc_article_convergence/campaign.py plan --shards 8
CAMPAIGN_ACCOUNT=<project>@h100 campaigns/gc_article_convergence/slurm/submit.sh 1 smoke   # first
CAMPAIGN_ACCOUNT=<project>@h100 campaigns/gc_article_convergence/slurm/submit.sh 8         # then everything
python campaigns/gc_article_convergence/campaign.py status
```

Each array index runs one shard: `campaign.py run --shard i/N`, in cluster mode. Cluster mode does not wait
for free GPU memory or for the workstation's GPU lock. The shard plan is deterministic: longest estimated
task first, to the least loaded shard. Resubmitting resumes, because completed tasks and completed runs are
skipped. A shard that exceeds the 20 h limit therefore continues where it stopped. Every shard writes
`markers/shard_<i>_of_<N>.complete` when it finishes.

## Running locally

Accuracy runs share the GPU with other jobs and never take the GPU lock. `--local` waits for 16 GiB of free
device memory before each run and pauses while a timing campaign holds the lock. Start the campaign detached,
so it survives the terminal or agent session:

```bash
OUT=run_outputs/campaigns/gc_article_convergence; mkdir -p "$OUT"
setsid nohup .venv/bin/python campaigns/gc_article_convergence/campaign.py run --shard 0/1 --group smoke --local \
  > "$OUT/driver_stdout.log" 2>&1 < /dev/null &
ps -o pgid= -p $!        # process group; stop with: kill -- -<pgid>
```

The completion marker is `$OUT/markers/shard_0_of_1.complete`.

## Outputs

Outputs go under `$HYBRIDGE_CAMPAIGN_OUT`, default `<repo>/run_outputs/campaigns/gc_article_convergence`:

- `runs/<task>/<family>/<run>.csv|json`: per-run rows, including the `*_accurate` errors and the exact-drift mode,
  with the final fields under `runs/<task>/<family>/fields/`. The run logs are under `runs/<task>/<family>/logs/`.
- `temporal_wave/`, `temporal_vortex/`, `exact_bdf3/`, `boundary_layer/`, `lattice/`, `smoke/`.
- `logs/<task>.log`, `markers/<task>.done|failed` (JSON: exit status, wall time, missing outputs).

After the runs, `campaign.py combine` does the following (no simulation):

- merges the BDF3 runs into `exact_bdf3/<family>_exact_drift.csv`;
- links every per-run CSV (smoke excluded) under `poisson_degree/<family>/` and merges them into
  `poisson_degree/poisson_degree.csv`, with orders per (family, p, q, variant);
- writes `projection_check.json`;
- writes `boundary_layer/analysis.txt`.

Pass criterion of the projection check: on every mesh of the group, the accurate density errors (Ω and the
interior square) of the projected DG(p + 2) drift agree with those of the pointwise exact drift within 1e-3
relative. The projection error of the drift is O(h^(p+3)), well below the density error.

## Bringing the results back

```bash
rsync -a <cluster>:<repo>/run_outputs/campaigns/gc_article_convergence/ \
  ~/src/gc-hdg-article/run_outputs/campaign_gc_article_convergence/
ln -s ~/src/gc-hdg-article/run_outputs/campaign_gc_article_convergence \
  <local repo>/run_outputs/article/campaign_gc_article_convergence
```

The table scripts (`scripts/guiding_center/studies/convergence/make_convergence_tables.py` and the article's table
builders) must then read the `*_accurate` columns and this layout (`poisson_degree/poisson_degree.csv`,
`runs/<task>/`, `temporal_*`). That change follows the campaign.

## Commit list

`python campaigns/gc_article_convergence/campaign.py deps` lists every file the campaign imports or reads: the
static import closure of the entry scripts, the AMGX configurations they name, this folder, and the tests of the
quadrature-point coefficient change. Each file comes with its git status. Commit every modified or untracked
file it lists. `CODEMAP.md` and `AGENTS.md` are machine-specific and never committed.
