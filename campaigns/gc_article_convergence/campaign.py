#!/usr/bin/env python3
"""GC article convergence campaign: every Section 5 / App. B-C GPU convergence run, with accurate errors.

Reruns the spatial ladders of the guiding-center article (k = 8 pi travelling wave, IMEX-ARK3, T = 1/16, the
article time steps), the raised-degree Poisson study, the compatible and exact drifts, the BDF3 exact-drift
cross-check, the two temporal studies and the diagnostics, so that every density error of the article comes from
runs that record the accurate error rule (2p + 2 collapsed-Gauss points per direction, ``*_accurate`` columns)
and save their final fields. The exact drift is the pointwise one (exact values at the transport's quadrature
points); the projected DG(p + 2) drift is the cross-check (group ``exact-drift-projection-check``).

One task is one independent ladder, run by one process (``run_poisson_degree_study.py`` for the degree study,
which itself starts one worker per run). Tasks are sharded deterministically by estimated cost
(``cost_estimates.json``) for a SLURM array (``slurm/``) or run locally. A task is done when every per-run output
it expects exists (the driver exits 0 even when a run fails); completed runs are skipped on resume.

Commands:
  list [--group G]                 tasks, runs and estimated hours
  plan --shards N [--group G]      deterministic assignment of the tasks to N shards
  status [--group G]               done / failed / partial / pending per task
  run --shard i/N [--group G] [--local] [--dry-run]
                                   run the tasks of shard i (``--local``: wait for free GPU memory and while a
                                   timing campaign holds the GPU lock, as on the workstation)
  combine                          merge the rows (poisson_degree.csv, BDF3 tables), projection check,
                                   boundary-layer analysis
  verify-meshes [--group G]        check the mesh cache against mesh_manifest.json
  deps                             files the campaign imports or reads, with their git status (commit list)
  costs                            regenerate cost_estimates.json from recorded local wall times

Environment: ``HYBRIDGE_PYTHON`` (interpreter of the runs, default: this one), ``HYBRIDGE_CAMPAIGN_OUT``
(outputs, default ``<repo>/run_outputs/campaigns/gc_article_convergence``).
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import meshes  # noqa: E402  (campaign folder)
from scripts.guiding_center.studies.convergence import run_poisson_degree_study as study  # noqa: E402

PYTHON = os.environ.get("HYBRIDGE_PYTHON") or sys.executable
OUT = Path(os.environ.get("HYBRIDGE_CAMPAIGN_OUT") or REPO / "run_outputs" / "campaigns" / "gc_article_convergence")
DT_TABLE = HERE / "dt_table.json"
DT_TABLE_BDF3 = HERE / "dt_table_exact_bdf3.json"
COSTS = HERE / "cost_estimates.json"
DRIVER = "scripts.guiding_center.studies.convergence.run_poisson_degree_study"
CONVERGENCE = "scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence"
BDF3 = "scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py"
SNAPSHOT = "scripts/guiding_center/studies/convergence/snapshot_manufactured_solution.py"
FINAL_TIME = 0.0625
AMPLITUDE = "A=0.028134884879909"  # 1 / (8 pi sqrt 2): unit crest-parallel drift
WAVE = ("--case-param", "kx=8*pi", "--case-param", "U=1", "--case-param", AMPLITUDE)
KY = {"aligned": "ky=8*pi", "misaligned": "ky=-8*pi"}
# Extended structured ladders (article ladders plus p = 3 at h = 0.0025 and p = 4 at h = 0.005).
STRUCTURED_EXTENDED = {2: (0.04, 0.02, 0.01, 0.005), 3: (0.04, 0.02, 0.01, 0.005, 0.0025),
                       4: (0.04, 0.02, 0.01, 0.005)}
DRIFTS = ("raw", "compatible", "compatible-primal", "exact")
PROJECTION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Task:
    """One independent command of the campaign.

    ``args``: arguments after the interpreter; ``local_args`` / ``cluster_args`` are appended in local or cluster
    mode. A degree-study task (``degree`` = the driver's plan options) knows its runs from the driver's ``plan``;
    every other task lists its ``outputs`` (relative to the output directory). ``meshes``: ``family:h`` keys of
    the cached Gmsh meshes it reads.
    """

    name: str
    group: str
    args: tuple[str, ...]
    meshes: tuple[str, ...] = ()
    gpu: bool = True
    outputs: tuple[str, ...] = ()
    degree: dict | None = None
    local_args: tuple[str, ...] = ()
    cluster_args: tuple[str, ...] = ()
    note: str = ""

    def command(self, *, local: bool) -> list[str]:
        """Interpreter and arguments of this task."""
        return [PYTHON, *self.args, *(self.local_args if local else self.cluster_args)]


def _sizes(values) -> str:
    return ",".join(f"{float(value):g}" for value in values)


def _gmsh_keys(family: str, sizes) -> tuple[str, ...]:
    """Mesh-cache keys of a ladder (structured meshes are not cached)."""
    mesh_family = study.FAMILIES[family]["mesh_family"]
    if mesh_family == "structured":
        return ()
    return tuple(meshes.key(mesh_family, size) for size in sizes)


def degree_task(name, group, family, order, offset, drift="raw", *, sizes=None, checks=True, mode="pointwise",
                steps=None, solver_rtol=None, note="") -> Task:
    """A ``run_poisson_degree_study.py`` invocation for one (family, p, q, drift) ladder."""
    ladder = study.LADDERS[study.FAMILIES[family]["mesh_family"]]
    sizes = tuple(sizes) if sizes is not None else tuple(ladder[order])
    output = OUT / "runs" / name
    args = ["-u", "-m", DRIVER, "--families", family, "--orders", str(order), "--offsets", str(offset),
            "--drift", drift, "--mesh-sizes", _sizes(sizes), "--dt-table", str(DT_TABLE),
            "--output-dir", str(output), "--baseline-dir", str(OUT / "no_baseline"),
            "--exact-drift-dir", str(OUT / "no_exact"), "-v", "1", "--log-every", "10"]
    if drift == "exact" and mode != "pointwise":
        args += ["--exact-drift-mode", mode]
    if checks:
        args.append("--checks")
    if steps is not None:
        args += ["--steps", str(steps)]
    if solver_rtol is not None:
        args += ["--solver-rtol", f"{solver_rtol:g}"]
    variant = {"drift": drift, "steps": steps, "solver_rtol": solver_rtol,
               "exact_drift_mode": mode if drift == "exact" else "pointwise"}
    return Task(name, group, tuple(args), meshes=_gmsh_keys(family, sizes),
                degree=dict(families=[family], orders=[order], offsets=[offset], mesh_sizes=list(sizes),
                            checks=checks, variant=variant, output=str(output)),
                local_args=("--min-free-gpu-mib", "16384"), cluster_args=("--ignore-gpu-lock", "--min-free-gpu-mib", "0"),
                note=note)


def _wave(family: str) -> tuple[str, ...]:
    return (*WAVE, "--case-param", KY[family])


def bdf3_task(family: str, order: int, size: float, *, dt=None, final_time=FINAL_TIME, group="exact-bdf3",
              name=None, output=None) -> Task:
    """One ``exact_drift_transport_rates.py`` run (BDF3, exact drift at the quadrature points, no Poisson)."""
    output = output or f"exact_bdf3/runs/{family}_p{order}_h{size:g}.csv"
    timing = ("--dt", repr(dt)) if dt is not None else ("--dt-table", str(DT_TABLE_BDF3))
    args = ("-u", BDF3, "--orders", str(order), "--mesh-sizes", f"{size:g}", "--mesh-family", "structured",
            *_wave(family), "--final-time", repr(final_time), *timing, "--output", str(OUT / output),
            "-v", "1", "--log-every", "10")
    return Task(name or f"exact-bdf3-{family}-p{order}-h{size:g}", group, args, outputs=(output,))


def snapshot_task(name, group, order, size, offset, dt, final_time, resolution, output) -> Task:
    """One raster snapshot run (``snapshot_manufactured_solution.py``), aligned structured mesh."""
    args = ("-u", SNAPSHOT, "--order", str(order), "--mesh-size", f"{size:g}", "--dt", repr(dt),
            "--final-time", repr(final_time), *_wave("aligned"), "--backend", "device", "--device-profile", "bsr",
            "--mesh-family", "structured", "--poisson-order-offset", str(offset), "--resolution", str(resolution),
            "--output", str(OUT / output))
    return Task(name, group, args, outputs=(output,))


def temporal_task(name, group, options: tuple[str, ...], mesh_key: str, output_dir: str, prefix: str) -> Task:
    """One temporal study of ``run_guiding_center_temporal_convergence.py`` (``--study manufactured``)."""
    args = ("-u", "-m", CONVERGENCE, "--study", "manufactured", *options, "--output-dir", str(OUT / output_dir),
            "--prefix", prefix, "--verbosity", "0")
    return Task(name, group, args, meshes=(mesh_key,), outputs=(f"{output_dir}/{prefix}.csv",))


def build_tasks() -> list[Task]:
    """The campaign's task registry, in group order."""
    tasks: list[Task] = []
    add = tasks.append
    # smoke: every code path for a few steps on the coarse meshes; run first on a new machine.
    for drift, offset, mode, label in (("raw", 0, "pointwise", "raw"), ("raw", 1, "pointwise", "raised"),
                                       ("compatible", 0, "pointwise", "compatible"),
                                       ("compatible-primal", 0, "pointwise", "compatible-primal"),
                                       ("exact", 0, "pointwise", "exact"), ("exact", 0, "projected", "exact-projected")):
        add(degree_task(f"smoke-gmsh-{label}", "smoke", "gmsh", 2, offset, drift, sizes=(0.04,), checks=False,
                        mode=mode, steps=3))
    add(degree_task("smoke-aligned-exact", "smoke", "aligned", 2, 0, "exact", sizes=(0.04,), checks=False, steps=3))
    add(degree_task("smoke-unstructured-raw", "smoke", "unstructured", 2, 0, sizes=(0.04,), checks=False, steps=3))
    add(temporal_task("smoke-temporal-wave", "smoke", (
        "--case", "rho_helm_wave", *WAVE, "--case-param", KY["aligned"], "--scheme", "imex-ark3,si-bdf2",
        "--final-time", "0.00390625", "--dts", "0.001953125", "--reference-dt", "0.0009765625",
        "--reference-scheme", "imex-ark3", "--mesh-size", "0.04", "--order", "2", "--poisson-tau", "1",
        "--backend", "device", "--device-profile", "bsr"), "gmsh:0.04", "smoke/temporal_wave", "smoke"))
    add(temporal_task("smoke-temporal-vortex", "smoke", (
        "--case", "orbiting_vortex", "--scheme", "imex-ark3", "--final-time", "0.2", "--dts", "0.1",
        "--reference-dt", "0.05", "--reference-scheme", "imex-ark3", "--error-quad-1d", "16", "--mesh-size", "0.025",
        "--order", "2", "--poisson-tau", "1", "--backend", "device", "--device-profile", "bsr"),
        "polygon:0.025", "smoke/temporal_vortex", "smoke"))
    add(bdf3_task("misaligned", 2, 0.04, dt=0.001953125, final_time=4 * 0.001953125, group="smoke",
                  name="smoke-bdf3", output="smoke/bdf3_misaligned_p2_h0.04.csv"))
    add(snapshot_task("smoke-snapshot", "smoke", 2, 0.04, 0, 0.001953125, 3 * 0.001953125, 200,
                      "smoke/snapshot_aligned_p2_h0.04.npz"))
    # lattice: edge orientation of the Gmsh meshes (no simulation, no GPU).
    for family in ("gmsh", "unstructured"):
        ladder = study.LADDERS[family]
        sizes = sorted({size for values in ladder.values() for size in values}
                       | (set(study.EXTRA_GMSH_SIZES) if family == "gmsh" else set()), reverse=True)
        add(Task(f"lattice-{family}", "lattice", ("-u", "-m", DRIVER, "--lattice", "--families", family,
                                                   "--output-dir", str(OUT / "lattice")),
                 meshes=tuple(meshes.key(family, size) for size in sizes), gpu=False,
                 outputs=(f"lattice/{family}_lattice.json",)))
    # Gmsh ladders (article Section 5): raw q = p, raised q = p + 1 and p + 2, compatible, compatible-primal, exact.
    for order in (2, 3, 4, 5, 6):
        add(degree_task(f"gmsh-raw-p{order}", "gmsh-raw", "gmsh", order, 0))
        for offset in (1, 2):
            add(degree_task(f"gmsh-raised-p{order}-s{offset}", "gmsh-raised", "gmsh", order, offset))
        for drift in ("compatible", "compatible-primal", "exact"):
            add(degree_task(f"gmsh-{drift}-p{order}", f"gmsh-{drift}", "gmsh", order, 0, drift))
    # The p = 2 showcase mesh (2,950,142 triangles): no dt/2 run.
    for drift, offset, label in (("raw", 0, "raw"), ("raw", 1, "raised"), ("compatible", 0, "compatible"),
                                 ("compatible-primal", 0, "compatible-primal"), ("exact", 0, "exact")):
        add(degree_task(f"gmsh-finest-{label}", "gmsh-finest", "gmsh", 2, offset, drift, sizes=(0.00177,),
                        checks=False))
    # Structured ladders (App. C): aligned and misaligned.
    for family in ("aligned", "misaligned"):
        for order, sizes in STRUCTURED_EXTENDED.items():
            add(degree_task(f"structured-raw-{family}-p{order}", "structured-raw", family, order, 0, sizes=sizes))
            add(degree_task(f"structured-raised-{family}-p{order}-s1", "structured-raised", family, order, 1,
                            sizes=sizes))
            add(degree_task(f"structured-raised-{family}-p{order}-s2", "structured-raised", family, order, 2))
            add(degree_task(f"structured-exact-{family}-p{order}", "structured-exact", family, order, 0, "exact",
                            sizes=sizes))
    for drift in ("compatible", "compatible-primal"):
        for order, sizes in STRUCTURED_EXTENDED.items():
            add(degree_task(f"structured-{drift}-misaligned-p{order}", f"structured-{drift}", "misaligned", order, 0,
                            drift, sizes=sizes))
    # Unstructured Gmsh (Delaunay) ladders, p = 2, 3, 4.
    for order in (2, 3, 4):
        for offset in (0, 1):
            add(degree_task(f"unstructured-raw-p{order}-s{offset}", "unstructured-raw", "unstructured", order, offset))
        for drift in ("compatible", "compatible-primal", "exact"):
            add(degree_task(f"unstructured-{drift}-p{order}", f"unstructured-{drift}", "unstructured", order, 0,
                            drift))
    # Cross-check of the pointwise exact drift: its DG(p + 2) projection on the two finest meshes of each ladder.
    for family, pairs in (("gmsh", {2: (0.005, 0.0025), 3: (0.01, 0.005), 4: (0.01, 0.005), 5: (0.02, 0.01),
                                    6: (0.04, 0.02)}),
                          ("misaligned", {2: (0.01, 0.005), 3: (0.005, 0.0025), 4: (0.01, 0.005)})):
        for order, sizes in pairs.items():
            add(degree_task(f"exact-projected-{family}-p{order}", "exact-drift-projection-check", family, order, 0,
                            "exact", sizes=sizes, checks=False, mode="projected"))
    # BDF3 exact-drift transport without Poisson (the earlier q = infinity rows), structured meshes.
    for family in ("aligned", "misaligned"):
        for order, size in ((2, 0.01), (2, 0.005), (3, 0.01), (4, 0.02)):
            add(bdf3_task(family, order, size))
    # Temporal studies: the wave on one Gmsh mesh, the orbiting vortex in the cross polygon (exact errors).
    add(temporal_task("temporal-wave", "temporal", (
        "--case", "rho_helm_wave", *WAVE, "--case-param", KY["aligned"], "--scheme", "imex-ark3,si-bdf2",
        "--final-time", repr(FINAL_TIME), "--dts", "0.00390625,0.001953125,0.0009765625,0.00048828125,0.000244140625",
        "--reference-dt", repr(FINAL_TIME / 2048), "--reference-scheme", "imex-ark3", "--mesh-size", "0.0125",
        "--order", "6", "--poisson-tau", "1", "--backend", "device", "--device-profile", "bsr"),
        "gmsh:0.0125", "temporal_wave", "hard_temporal"))
    add(temporal_task("temporal-vortex", "temporal", (
        "--case", "orbiting_vortex", "--scheme", "si-euler,si-bdf2,imex-ark3", "--final-time", "0.5",
        "--dts", "0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125", "--reference-dt", "0.0001953125",
        "--reference-scheme", "imex-ark3", "--error-quad-1d", "16", "--mesh-size", "0.00625", "--order", "6",
        "--poisson-tau", "1", "--backend", "device", "--device-profile", "bsr"),
        "polygon:0.00625", "temporal_vortex", "temporal"))
    # Diagnostics: error floor with tighter AMGX tolerances, boundary-layer snapshots of the aligned mesh.
    add(degree_task("floor-gmsh-p5-s0-h0.01", "diagnostics", "gmsh", 5, 0, sizes=(0.01,), checks=False,
                    solver_rtol=1e-13))
    add(degree_task("floor-gmsh-p4-s1-h0.005", "diagnostics", "gmsh", 4, 1, sizes=(0.005,), checks=False,
                    solver_rtol=1e-13))
    for offset in (0, 1):
        add(snapshot_task(f"boundary-layer-p3-q{3 + offset}", "diagnostics", 3, 0.005, offset,
                          float(json.loads(DT_TABLE.read_text())["3,0.005"]), FINAL_TIME, 2000,
                          f"boundary_layer/aligned_p3_q{3 + offset}_h0.005.npz"))
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate task names")
    return tasks


GROUP_PURPOSE = {
    "smoke": "every code path, a few steps on coarse meshes (run first on a new machine)",
    "lattice": "edge-lattice orientation of the Gmsh meshes (no simulation)",
    "gmsh-raw": "Gmsh ladders, raw drift q = p (article Table tab:space-convergence)",
    "gmsh-raised": "Gmsh ladders, raised Poisson degree q = p + 1, p + 2",
    "gmsh-compatible": "Gmsh ladders, compatible drift V2 (Oswald average of the p + 1 postprocessed potential)",
    "gmsh-compatible-primal": "Gmsh ladders, Oswald average of phi_h itself (averaging without the postprocess)",
    "gmsh-exact": "Gmsh ladders, pointwise exact drift (q = infinity reference with the coupled scheme)",
    "gmsh-finest": "p = 2 on 2,950,142 triangles: raw, q = p + 1, compatible, compatible-primal, exact",
    "structured-raw": "structured aligned/misaligned ladders, raw q = p (extended ladders)",
    "structured-raised": "structured ladders, q = p + 1 (extended) and q = p + 2 (article ladders)",
    "structured-exact": "structured ladders, pointwise exact drift",
    "structured-compatible": "misaligned ladders, compatible drift V2",
    "structured-compatible-primal": "misaligned ladders, Oswald average of phi_h",
    "unstructured-raw": "Gmsh Delaunay ladders, raw q = p and q = p + 1",
    "unstructured-compatible": "Gmsh Delaunay ladders, compatible drift V2",
    "unstructured-compatible-primal": "Gmsh Delaunay ladders, Oswald average of phi_h",
    "unstructured-exact": "Gmsh Delaunay ladders, pointwise exact drift",
    "exact-drift-projection-check": "projected DG(p + 2) exact drift on the two finest meshes: must match the "
                                    "pointwise one within 1e-3",
    "exact-bdf3": "BDF3 transport with the exact drift and no Poisson solve (earlier q = infinity rows)",
    "temporal": "temporal convergence: wave (p = 6, Gmsh) and orbiting vortex (p = 6, cross polygon)",
    "diagnostics": "error floor at AMGX rtol 1e-13; boundary-layer snapshots of the aligned mesh",
}


# --------------------------------------------------------------------------------------------
# Expected outputs and state
# --------------------------------------------------------------------------------------------

def _dt_table() -> dict:
    return json.loads(DT_TABLE.read_text())


def degree_runs(task: Task) -> list:
    """The driver's runs of a degree-study task (``Run`` objects)."""
    options = task.degree
    variant = {key: value for key, value in options["variant"].items() if value is not None}
    return study.plan(options["families"], options["orders"], options["offsets"], options["mesh_sizes"],
                      _dt_table(), options["checks"], **variant)


def expected_outputs(task: Task) -> list[Path]:
    """Per-run outputs a task must produce."""
    if task.degree is not None:
        output = Path(task.degree["output"])
        return [study._result_path(output, run) for run in degree_runs(task)]
    return [OUT / path for path in task.outputs]


def task_state(task: Task) -> tuple[str, int, int]:
    """``(state, present, expected)``; state is done, failed, partial or pending."""
    outputs = expected_outputs(task)
    present = sum(path.exists() for path in outputs)
    if present == len(outputs):
        return "done", present, len(outputs)
    if (OUT / "markers" / f"{task.name}.failed").exists():
        return "failed", present, len(outputs)
    return ("partial" if present else "pending"), present, len(outputs)


def _select(tasks: list[Task], groups: str | None) -> list[Task]:
    if not groups:
        return tasks
    wanted = {group.strip() for group in groups.split(",") if group.strip()}
    unknown = wanted - {task.group for task in tasks}
    if unknown:
        raise SystemExit(f"unknown groups {sorted(unknown)}; choose from {sorted(GROUP_PURPOSE)}")
    return [task for task in tasks if task.group in wanted]


# --------------------------------------------------------------------------------------------
# Costs and sharding
# --------------------------------------------------------------------------------------------

def _triangles(family: str, size: float) -> int:
    mesh_family = study.FAMILIES[family]["mesh_family"] if family in study.FAMILIES else family
    if mesh_family == "structured":
        return 2 * int(round(2.0 / size)) ** 2
    entry = meshes.load_manifest().get(meshes.key(mesh_family, size))
    if entry is not None:
        return int(entry["triangles"])
    return int(round(2.31 * 4.0 / (size * size)))  # Gmsh: about 2.3 triangles per h^2 on (-1, 1)^2


def _recorded_rows() -> list[dict]:
    """Recorded local runs (article and earlier campaigns): mesh family, degrees, drift, size, steps, wall time."""
    article = REPO / "run_outputs" / "article"
    sources = [
        ("convergence_poisson_degree/gmsh/*.csv", "gmsh"), ("convergence_poisson_degree/aligned/*.csv", "structured"),
        ("convergence_poisson_degree/misaligned/*.csv", "structured"),
        ("convergence_poisson_degree/compatible_device/gmsh/*.csv", "gmsh"),
        ("convergence_poisson_degree/compatible_device/misaligned/*.csv", "structured"),
        ("convergence_hard/spatial/hard_*.csv", "gmsh"), ("convergence_hard/structured/*.csv", "structured"),
        ("convergence_unstructured/unstructured/*.csv", "unstructured"),
    ]
    rows = []
    for pattern, mesh_family in sources:
        for path in sorted(article.glob(pattern)):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        rows.append({"mesh_family": row.get("mesh_family") or mesh_family, "order": int(row["order"]),
                                     "poisson_order": int(float(row.get("poisson_order") or row["order"])),
                                     "drift": row.get("electric_field") or "raw", "triangles": int(row["triangles"]),
                                     "steps": int(row["num_steps"]), "wall": float(row["wall_time_s"])})
                    except (KeyError, TypeError, ValueError):
                        continue
    return rows


RUN_OVERHEAD_S = 60.0  # setup, JIT and the first Poisson solve of one worker
DEFAULT_RATE = 4e-7    # s per triangle-step at p = 2 when nothing is recorded


def _run_seconds(rows: list[dict], *, mesh_family, order, poisson_order, drift, triangles, steps) -> tuple[float, str]:
    """Estimated wall time of one run: a recorded run on the same mesh scaled by its step count, else a rate."""
    kind = "raw" if drift.startswith("exact") else drift
    same = [r["wall"] * steps / r["steps"] for r in rows
            if (r["mesh_family"], r["order"], r["poisson_order"], r["drift"]) == (mesh_family, order, poisson_order, kind)
            and abs(r["triangles"] - triangles) <= 0.01 * triangles]
    if same:
        return statistics.median(same), "recorded (same mesh)"
    rate, source = _rate(rows, mesh_family=mesh_family, order=order, poisson_order=poisson_order, drift=drift)
    return RUN_OVERHEAD_S + rate * steps * triangles, source


def _rate(rows: list[dict], *, mesh_family, order, poisson_order, drift) -> tuple[float, str]:
    """Seconds per triangle-step of the closest recorded configuration."""
    kind = "raw" if drift.startswith("exact") else drift
    for keys, label in (((mesh_family, order, poisson_order, kind), "recorded"),
                        ((None, order, poisson_order, kind), "recorded (other mesh family)"),
                        ((None, order, poisson_order, None), "recorded (other drift)"),
                        ((None, order, None, None), "recorded (other Poisson degree)")):
        family_key, order_key, q_key, drift_key = keys
        matching = sorted((r for r in rows
                           if r["order"] == order_key and (q_key is None or r["poisson_order"] == q_key)
                           and (family_key is None or r["mesh_family"] == family_key)
                           and (drift_key is None or r["drift"] == drift_key) and r["triangles"] >= 20000),
                          key=lambda r: r["triangles"])
        if matching:
            # The two largest recorded meshes: the cost per triangle-step falls as the GPU fills up.
            rates = [r["wall"] / (r["steps"] * r["triangles"]) for r in matching[-2:]]
            scale = 1.0 if q_key is not None else ((poisson_order + 1) / (order + 1)) ** 3
            return statistics.median(rates) * scale, label
    nel = (order + 1) * (order + 2) / 2
    return DEFAULT_RATE * (nel / 6.0) ** 2, "model"


def estimate_task(task: Task, rows: list[dict]) -> dict:
    """Estimated hours of one task from recorded rates (``rows``) or the model."""
    fixed = {"temporal-wave": 1.0, "temporal-vortex": 3.5}
    if task.name in fixed:
        return {"hours": fixed[task.name], "source": "recorded (whole study)", "runs": 1}
    if task.group in {"smoke", "lattice"}:
        return {"hours": 0.05 if task.group == "smoke" else 0.1, "source": "model", "runs": 1}
    if task.degree is None:  # BDF3 runs and snapshots: one coupled run on a structured mesh
        order = int(task.args[task.args.index("--order" if "--order" in task.args else "--orders") + 1])
        size = float(task.args[task.args.index("--mesh-size" if "--mesh-size" in task.args else "--mesh-sizes") + 1])
        dt_text = (task.args[task.args.index("--dt") + 1] if "--dt" in task.args
                   else json.loads(DT_TABLE_BDF3.read_text()).get(f"{order},{size:g}"))
        steps = math.ceil(FINAL_TIME / float(dt_text) - 1e-9)
        offset = int(task.args[task.args.index("--poisson-order-offset") + 1]) if "--poisson-order-offset" in task.args else 0
        seconds, source = _run_seconds(rows, mesh_family="structured", order=order, poisson_order=order + offset,
                                       drift="raw", triangles=_triangles("aligned", size), steps=steps)
        return {"hours": seconds / 3600.0, "source": source, "runs": 1}
    seconds, sources = 0.0, set()
    runs = degree_runs(task)
    for run in runs:
        mesh_family = study.FAMILIES[run.family]["mesh_family"]
        run_seconds, source = _run_seconds(
            rows, mesh_family=mesh_family, order=run.order, poisson_order=run.poisson_order, drift=run.drift,
            triangles=_triangles(run.family, run.mesh_size), steps=math.ceil(run.final_time / run.dt - 1e-9))
        seconds += run_seconds * (1.3 if run.solver_rtol is not None else 1.0)
        sources.add(source)
    return {"hours": seconds / 3600.0, "source": ", ".join(sorted(sources)), "runs": len(runs)}


def write_costs(tasks: list[Task]) -> Path:
    """Write ``cost_estimates.json`` (run on the machine holding the recorded runs, then commit it)."""
    rows = _recorded_rows()
    estimates = {task.name: estimate_task(task, rows) for task in tasks}
    COSTS.write_text(json.dumps({"recorded_rows": len(rows), "hardware": "local RTX PRO 5000 Blackwell records",
                                 "tasks": estimates}, indent=1) + "\n")
    print(f"{len(estimates)} tasks, {sum(e['hours'] for e in estimates.values()):.1f} estimated hours "
          f"from {len(rows)} recorded runs -> {COSTS}")
    return COSTS


def task_hours(tasks: list[Task]) -> dict[str, float]:
    """Estimated hours per task (``cost_estimates.json``; 1 h for a task it does not list)."""
    recorded = json.loads(COSTS.read_text())["tasks"] if COSTS.exists() else {}
    return {task.name: float(recorded.get(task.name, {}).get("hours", 1.0)) for task in tasks}


def shards(tasks: list[Task], count: int) -> list[list[Task]]:
    """Longest-processing-time assignment: largest task first to the least loaded shard (ties: lowest index)."""
    if count < 1:
        raise SystemExit("--shards must be at least 1")
    hours = task_hours(tasks)
    loads, assignment = [0.0] * count, [[] for _ in range(count)]
    for task in sorted(tasks, key=lambda task: (-hours[task.name], task.name)):
        index = min(range(count), key=lambda i: (loads[i], i))
        assignment[index].append(task)
        loads[index] += hours[task.name]
    return assignment


def _parse_shard(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if not match or int(match.group(1)) >= int(match.group(2)):
        raise SystemExit("--shard must be i/N with 0 <= i < N")
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------

def cmd_list(tasks: list[Task]) -> None:
    hours = task_hours(tasks)
    print(f"{'task':<46}{'group':<31}{'runs':>5}{'hours':>8}  meshes")
    for task in tasks:
        runs = len(expected_outputs(task))
        print(f"{task.name:<46}{task.group:<31}{runs:>5}{hours[task.name]:>8.2f}  {','.join(task.meshes) or '-'}")
    groups = {}
    for task in tasks:
        count, total = groups.get(task.group, (0, 0.0))
        groups[task.group] = (count + len(expected_outputs(task)), total + hours[task.name])
    print(f"\n{'group':<31}{'runs':>6}{'hours':>8}  purpose")
    for group, (count, total) in groups.items():
        print(f"{group:<31}{count:>6}{total:>8.2f}  {GROUP_PURPOSE.get(group, '')}")
    print(f"{len(tasks)} tasks, {sum(c for c, _ in groups.values())} runs, "
          f"{sum(t for _, t in groups.values()):.1f} estimated local GPU-hours")


def cmd_plan(tasks: list[Task], count: int) -> None:
    hours = task_hours(tasks)
    assignment = shards(tasks, count)
    for index, members in enumerate(assignment):
        total = sum(hours[task.name] for task in members)
        print(f"shard {index}/{count}: {len(members)} tasks, {total:.1f} h")
        for task in members:
            print(f"  {task.name:<46}{hours[task.name]:>7.2f} h")
    loads = [sum(hours[task.name] for task in members) for members in assignment]
    print(f"{len(tasks)} tasks, {sum(loads):.1f} h in total; largest shard {max(loads):.1f} h, smallest "
          f"{min(loads):.1f} h (estimates from local RTX PRO 5000 records; other GPUs differ)")


def cmd_status(tasks: list[Task]) -> None:
    counts = {}
    for task in tasks:
        state, present, expected = task_state(task)
        counts[state] = counts.get(state, 0) + 1
        print(f"{task.name:<46}{state:<9}{present:>4}/{expected:<4}")
    print("  ".join(f"{state}: {count}" for state, count in sorted(counts.items())) + f"  (outputs under {OUT})")


def _write_marker(task: Task, kind: str, payload: dict) -> None:
    markers = OUT / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / f"{task.name}.{kind}").write_text(json.dumps(payload, indent=1) + "\n")
    other = markers / f"{task.name}.{'failed' if kind == 'done' else 'done'}"
    if other.exists():
        other.unlink()


def run_task(task: Task, *, local: bool, dry_run: bool) -> str:
    """Run one task unless its outputs are complete; write its marker; return its state."""
    state, present, expected = task_state(task)
    command = task.command(local=local)
    if state == "done":
        print(f"[campaign] {task.name}: done ({present}/{expected}), skipped", flush=True)
        if not dry_run and not (OUT / "markers" / f"{task.name}.done").exists():
            _write_marker(task, "done", {"task": task.name, "note": "outputs present"})
        return "done"
    if dry_run:
        print(f"[campaign] {task.name}: {state} ({present}/{expected})\n  {shlex.join(command)}", flush=True)
        return state
    problems = meshes.verify(list(task.meshes), quiet=True) if task.meshes else []
    if problems:
        print(f"[campaign] {task.name}: mesh check failed: {'; '.join(problems)}", flush=True)
        _write_marker(task, "failed", {"task": task.name, "reason": "mesh check", "problems": problems})
        return "failed"
    log = OUT / "logs" / f"{task.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[campaign] {time.strftime('%F %T')} {task.name}: start ({present}/{expected} present), log {log}",
          flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[campaign] {time.strftime('%F %T')} {shlex.join(command)}\n")
        handle.flush()
        status = subprocess.call(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"})
    missing = [str(path.relative_to(OUT)) if OUT in path.parents else str(path)
               for path in expected_outputs(task) if not path.exists()]
    payload = {"task": task.name, "exit_status": status, "wall_s": time.time() - started,
               "finished": time.strftime("%F %T"), "missing": missing, "command": command}
    kind = "done" if status == 0 and not missing else "failed"
    _write_marker(task, kind, payload)
    print(f"[campaign] {time.strftime('%F %T')} {task.name}: {kind} (exit {status}, {len(missing)} missing, "
          f"{payload['wall_s'] / 3600:.2f} h)", flush=True)
    return kind


def cmd_run(tasks: list[Task], shard: str, *, local: bool, dry_run: bool) -> None:
    index, count = _parse_shard(shard)
    members = shards(tasks, count)[index]
    print(f"[campaign] shard {index}/{count}: {len(members)} tasks, outputs {OUT}, python {PYTHON}", flush=True)
    states = [run_task(task, local=local, dry_run=dry_run) for task in members]
    if not dry_run:
        summary = {state: states.count(state) for state in sorted(set(states))}
        marker = OUT / "markers" / f"shard_{index}_of_{count}.complete"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"finished": time.strftime("%F %T"), "states": summary,
                                      "tasks": [task.name for task in members]}, indent=1) + "\n")
        print(f"[campaign] shard {index}/{count} finished: {summary}", flush=True)
        if summary.get("failed"):
            raise SystemExit(1)


def _link_rows(tasks: list[Task], target: Path) -> int:
    """Symlink every per-run CSV of the degree tasks (smoke excluded) into ``target/<family>/``."""
    count = 0
    for task in tasks:
        if task.degree is None or task.group == "smoke":
            continue
        for path in expected_outputs(task):
            if not path.exists():
                continue
            link = target / path.parent.name / path.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(path, link.parent))
            count += 1
    return count


def _projection_check(tasks: list[Task]) -> dict:
    """Projected DG(p + 2) exact drift against the pointwise one: accurate density errors, same (family, p, h)."""
    def rows_of(task):
        for path in expected_outputs(task):
            if path.exists() and not path.stem.endswith("_dthalf"):
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        yield path.parent.name, int(row["order"]), float(row["mesh_size"]), row

    pointwise = {}
    for task in tasks:
        if task.degree is not None and task.group in {"gmsh-exact", "structured-exact"}:
            for family, order, size, row in rows_of(task):
                pointwise[family, order, size] = row
    comparisons, worst = [], 0.0
    for task in tasks:
        if task.group != "exact-drift-projection-check":
            continue
        for family, order, size, row in rows_of(task):
            reference = pointwise.get((family, order, size))
            if reference is None:
                comparisons.append({"family": family, "order": order, "mesh_size": size, "status": "no pointwise run"})
                continue
            entry = {"family": family, "order": order, "mesh_size": size}
            for key in ("rho_l2_error_accurate", "rho_l2_interior_error_accurate"):
                if row.get(key) and reference.get(key):
                    change = abs(float(row[key]) / float(reference[key]) - 1.0)
                    entry[key.replace("_error_accurate", "_relative_difference")] = change
                    worst = max(worst, change)
            comparisons.append(entry)
    complete = bool(comparisons) and all("status" not in entry for entry in comparisons)
    result = {"tolerance": PROJECTION_TOLERANCE, "largest_relative_difference": worst, "complete": complete,
              "passed": complete and worst <= PROJECTION_TOLERANCE, "comparisons": comparisons}
    (OUT / "projection_check.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def cmd_combine(tasks: list[Task]) -> None:
    """Merge the campaign's rows; no simulation."""
    OUT.mkdir(parents=True, exist_ok=True)
    for family in ("aligned", "misaligned"):
        parts = sorted((OUT / "exact_bdf3" / "runs").glob(f"{family}_p*_h*.csv"))
        if parts:
            subprocess.check_call([PYTHON, BDF3, "--combine", *map(str, parts), "--output",
                                   str(OUT / "exact_bdf3" / f"{family}_exact_drift.csv")], cwd=REPO)
    merged = OUT / "poisson_degree"
    count = _link_rows(tasks, merged)
    print(f"[campaign] {count} per-run CSVs linked under {merged}")
    if count:
        subprocess.check_call([PYTHON, "-m", DRIVER, "--combine-only", "--families",
                               "gmsh,aligned,misaligned,unstructured", "--orders", "2,3,4,5,6",
                               "--output-dir", str(merged), "--baseline-dir", str(OUT / "no_baseline"),
                               "--exact-drift-dir", str(OUT / "exact_bdf3")], cwd=REPO)
    check = _projection_check(tasks)
    print(f"[campaign] exact-drift projection check: {'passed' if check['passed'] else 'NOT passed'} "
          f"(largest relative difference {check['largest_relative_difference']:.2e}, tolerance "
          f"{PROJECTION_TOLERANCE:g}, complete={check['complete']})")
    snapshots = sorted((OUT / "boundary_layer").glob("aligned_p3_q*_h0.005.npz"))
    if len(snapshots) == 2:
        with (OUT / "boundary_layer" / "analysis.txt").open("w", encoding="utf-8") as handle:
            subprocess.check_call([PYTHON, str(HERE / "analyze_boundary_layer.py"), *map(str, snapshots)],
                                  cwd=REPO, stdout=handle)
        print(f"[campaign] boundary-layer analysis -> {OUT / 'boundary_layer' / 'analysis.txt'}")


def cmd_verify_meshes(tasks: list[Task]) -> None:
    keys = sorted({key for task in tasks for key in task.meshes})
    if meshes.verify(keys):
        raise SystemExit(1)


# --------------------------------------------------------------------------------------------
# Dependencies (commit list)
# --------------------------------------------------------------------------------------------

ENTRY_SCRIPTS = (
    "scripts/guiding_center/studies/convergence/run_poisson_degree_study.py",
    "scripts/guiding_center/studies/convergence/run_guiding_center_temporal_convergence.py",
    "scripts/guiding_center/studies/convergence/snapshot_manufactured_solution.py",
    "scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py",
    "scripts/guiding_center/runtime/exact_drift.py",
    "scripts/guiding_center/runtime/compatible_drift.py",
)
CHANGE_TESTS = (
    "tests/test_quadrature_coefficients.py", "tests/test_transport_quadrature_coefficients.py",
    "tests/test_transport_quadrature_convergence.py", "tests/test_guiding_center_exact_drift.py",
    "tests/test_transport_cross_space_advection.py",
)
NEVER_COMMIT = {"CODEMAP.md", "AGENTS.md"}


def _module_file(name: str) -> Path | None:
    parts = name.split(".")
    for count in range(len(parts), 0, -1):
        base = REPO.joinpath(*parts[:count])
        if base.with_suffix(".py").exists():
            return base.with_suffix(".py")
        if (base / "__init__.py").exists():
            return base / "__init__.py"
    return None


def _imports(path: Path) -> set[str]:
    """Repository modules named by every import statement of ``path`` (function-level ones included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = ".".join(path.relative_to(REPO).parent.parts)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.split(".")[0] in {"hybridge", "scripts", "projects"}}


def import_closure(entries) -> set[Path]:
    """Static import closure of ``entries`` (repository files only, package ``__init__`` files included)."""
    seen, todo = set(), [REPO / entry for entry in entries]
    while todo:
        path = todo.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for parent in path.relative_to(REPO).parents:
            init = REPO / parent / "__init__.py"
            if init.exists() and init not in seen:
                todo.append(init)
        if path.suffix == ".py":
            for module in _imports(path):
                found = _module_file(module)
                if found is not None and found not in seen:
                    todo.append(found)
    return seen


def cmd_deps() -> None:
    closure = import_closure(ENTRY_SCRIPTS)
    configs = set()
    amgx = REPO / "configs" / "amgx"
    for path in closure:
        for name in re.findall(r"[A-Za-z0-9_.-]+\.json", path.read_text(encoding="utf-8")):
            if (amgx / name).is_file():
                configs.add(amgx / name)
    campaign = {path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    files = sorted({str(path.relative_to(REPO)) for path in closure | configs | campaign}
                   | set(CHANGE_TESTS))
    status = {}
    listing = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    for line in listing.splitlines():
        status[line[3:].strip()] = line[:2]
    groups = {"modified": [], "untracked": [], "unchanged": []}
    for name in files:
        if Path(name).name in NEVER_COMMIT:
            continue
        code = status.get(name, "")
        groups["untracked" if code == "??" else "modified" if code.strip() else "unchanged"].append(name)
    print(f"{len(files)} files (import closure of the entry scripts: {len(closure)}, AMGX configs: {len(configs)}, "
          f"campaign folder: {len(campaign)}, tests: {len(CHANGE_TESTS)}); never commit {sorted(NEVER_COMMIT)}")
    for kind in ("modified", "untracked"):
        print(f"{kind.upper()} ({len(groups[kind])}): commit these")
        print("\n".join(f"  {name}" for name in groups[kind]))
    print(f"UNCHANGED ({len(groups['unchanged'])}): already committed")


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "status", "verify-meshes"):
        commands.add_parser(name).add_argument("--group", default=None, help="comma list of groups")
    planner = commands.add_parser("plan")
    planner.add_argument("--shards", type=int, required=True)
    planner.add_argument("--group", default=None)
    runner = commands.add_parser("run")
    runner.add_argument("--shard", required=True, help="i/N")
    runner.add_argument("--group", default=None)
    runner.add_argument("--local", action="store_true",
                        help="workstation mode: wait for 16 GiB of free GPU memory and while a timing campaign "
                             "holds the GPU lock (cluster mode: neither)")
    runner.add_argument("--dry-run", action="store_true", help="print the commands and states only")
    commands.add_parser("combine")
    commands.add_parser("deps")
    commands.add_parser("costs")
    args = parser.parse_args()
    tasks = build_tasks()
    selected = _select(tasks, getattr(args, "group", None))
    if args.command == "list":
        cmd_list(selected)
    elif args.command == "plan":
        cmd_plan(selected, args.shards)
    elif args.command == "status":
        cmd_status(selected)
    elif args.command == "run":
        cmd_run(selected, args.shard, local=args.local, dry_run=args.dry_run)
    elif args.command == "combine":
        cmd_combine(tasks)
    elif args.command == "verify-meshes":
        cmd_verify_meshes(selected)
    elif args.command == "deps":
        cmd_deps()
    elif args.command == "costs":
        write_costs(tasks)


if __name__ == "__main__":
    main()
