#!/usr/bin/env python3
r"""Raised-order Poisson study: does a better-resolved drift restore the h^{p+1} density rate?

With the raw degree-p LDG-H drift, the density of the article's k = 8 pi travelling wave
converges at about p+1/2 on misaligned structured meshes (article Table tab:alignment). With the exact drift
(``exact_drift_transport_rates.py``) it converges at p+1. This study keeps the coupled problem and
only raises the Poisson degree to q = p + s. The transport uses the raw degree-q flux through
cross-space drift tables, without projection into DG(p) (``hybridge.transport.advection_field``). If the
order recovers, the under-resolved electric field limits the density order.

Everything else is the configuration of the article runs (``run_outputs/article/convergence_hard``):
IMEX-ARK3, T = 1/16, Poisson tau = 1, conflict-averaged upwind, device profile ``bsr`` (raw-CUDA
assembly, AMGX at rtol 1e-11), Δt from ``convergence_hard/dt_table.json``. Families:

* ``misaligned``: structured meshes, k_y = -k_x (Table tab:alignment);
* ``aligned``: structured meshes, k_y = k_x (Table tab:alignment);
* ``gmsh``: Gmsh meshes, k_y = k_x (Section 5 ladder, Table tab:space-convergence; q up to 8 at p = 6);
* ``unstructured``: Gmsh Delaunay meshes (runner mesh family ``unstructured``: no lattice), k_y = k_x, the
  Gmsh ladders of p = 2, 3, 4. It has no article baseline, so the study runs offset 0 itself (own
  ``--output-dir``), with the raw drift (``--offsets 0,1``) and with ``--drift compatible``.

Offset 0 is not rerun for the families with an article baseline: its rows are the article runs
(``--baseline-dir``). Offset-0 runs on meshes outside the article ladder (``--mesh-sizes``) extend it: in the
article directory they are allowed, and ``--combine-only`` merges them with the baseline rows. The exact-drift rows
(``--exact-drift-dir``, ``<family>_exact_drift.csv``) are the q = infinity limit on the structured meshes.

Each run is one worker process (AMGX device memory accumulates within a process). This is an
accuracy study, so runs share the GPU with other jobs and take no lock (``--gpu-lock`` is for timing
studies): each run waits until ``--min-free-gpu-mib`` of device memory is free (nvidia-smi), and its
peak device memory is logged with its exit status. Its output is streamed line by line to the terminal,
``<output>/<family>/logs/<run>.log`` and ``<output>/campaign.log``, with an ETA appended to the
runner's step summaries. ``-v`` follows the guiding-center levels: 0 quiet, 1 step summaries
(time, conservation, solver iterations and times, density/potential errors) every ``--log-every``
steps, 2 adds solver phase logs, 3 logs every timing. Completed runs are skipped unless
``--rerun``. Finally the rows are merged into ``<output>/poisson_degree.csv`` with orders per
(family, p, q), and the dt/2 checks (``--checks``) are compared with their full-dt runs.

Other configurations (own ``--output-dir``; they tag the run names, ``Run.variant``):
``--drift compatible`` (V2: Oswald average of the p+1 postprocessed potential, offset 0; the raw-CUDA
Poisson postprocesses on the GPU), ``--drift compatible-primal`` (Oswald average of phi_h itself, a
compatible drift of degree p-1: the averaging without the postprocess, offset 0), ``--drift exact`` (the exact drift at
every solve time: the exact-drift reference with the scheme and time steps of the coupled runs; offset 0; by default
its values at the transport's quadrature points, tag ``_exact``; ``--exact-drift-mode projected`` uses the L2
projection of the exact flux onto DG(p+s) instead, s = ``--exact-drift-order-offset`` (default 2), tag
``_exactp<s>`` for every s), ``--scheme``, ``--backend host`` (Numba + PyPardiso, SI schemes),
``--solver-rtol`` (relative tolerance of the AMGX Poisson and transport solves instead of 1e-11, absolute
tolerance a tenth of it; tag ``_rtol<value>``).
Timing runs: ``--steps N`` stops each run after N steps (median step time, first two steps excluded),
and ``--exclusive-gpu`` holds the GPU lock, waits until no other compute process runs on the GPU and
writes ``<log>.gpu.json`` recording whether one appeared during the run.

``--lattice`` (no simulation) measures the meshes of the Gmsh families in ``--families`` (default
``gmsh``) instead: in the interior square (-0.6, 0.6)^2 the ``gmsh`` meshes are lattices of equilateral
triangles whose orientation changes with the target size, which changes how many edge families the
drift is tangent to; the ``unstructured`` meshes are no lattice. It writes ``<output>/<family>_lattice.json``
(``hybridge.core.mesh.edge_lattice_orientation`` of every ladder mesh, for ``gmsh`` also of the
article's 2,950,142-triangle mesh, loaded from the mesh cache); ``gmsh_lattice.json`` is read by
``make_convergence_tables.py`` and ``plot_convergence_study.py`` (``--lattice``).
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from argparse import SUPPRESS, ArgumentParser
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ROOT = Path(__file__).resolve().parents[3]
K8 = 8.0 * math.pi
AMPLITUDE = 1.0 / (K8 * math.sqrt(2.0))
FINAL_TIME = 0.0625
FAMILIES = {
    "misaligned": {"mesh_family": "structured", "ky": -K8, "baseline": "structured/misaligned_p{p}.csv"},
    "aligned": {"mesh_family": "structured", "ky": K8, "baseline": "structured/aligned_p{p}.csv"},
    "gmsh": {"mesh_family": "gmsh", "ky": K8, "baseline": "spatial/hard_p{p}.csv"},
    # No article baseline: the q = p rows are this study's own offset-0 runs.
    "unstructured": {"mesh_family": "unstructured", "ky": K8, "baseline": None},
}
# The article ladders (convergence_hard/run_aligned.py and run_ladders.py).
LADDERS = {
    "structured": {2: (0.04, 0.02, 0.01, 0.005), 3: (0.04, 0.02, 0.01, 0.005), 4: (0.04, 0.02, 0.01)},
    "gmsh": {2: (0.04, 0.02, 0.01, 0.005, 0.0025), 3: (0.04, 0.02, 0.01, 0.005),
             4: (0.08, 0.04, 0.02, 0.01, 0.005), 5: (0.08, 0.04, 0.02, 0.01), 6: (0.08, 0.04, 0.02)},
}
LADDERS["unstructured"] = {order: LADDERS["gmsh"][order] for order in (2, 3, 4)}
# Gmsh sizes reported in the article besides the ladders: the p = 2 run on 2,950,142 triangles
# (convergence_hard/spatial/hard_showcase_p2.csv).
EXTRA_GMSH_SIZES = (0.00177,)
LATTICE_BOX = ((-0.6, 0.6), (-0.6, 0.6))
ARTICLE = ROOT / "run_outputs" / "article"
DEFAULT_OUTPUT = ARTICLE / "convergence_poisson_degree"
DEFAULT_BASELINE = ARTICLE / "convergence_hard"
DEFAULT_EXACT = ARTICLE / "convergence_exact_drift"
GPU_LOCK = ROOT / ".cache" / "gpu-campaign.lock"
# Step labels of the runner: "[gc] step=00010/00137" (v1), "| step 00010/00137 |" (v2), "[gc:linear] step" (v3).
STEP_LINE = re.compile(r"\bstep[= ](\d+)/(\d+)")
REPORT_COLUMNS = (
    "family", "source", "variant", "order", "poisson_order", "mesh_size", "triangles", "dt", "num_steps",
    "rho_l2_interior_error", "rho_l2_interior_rate", "rho_l2_error", "rho_l2_rate", "phi_l2_error",
    "phi_l2_rate", "step_time_median_s", "wall_time_s", "trace_unknowns", "poisson_trace_unknowns",
)


@dataclass(frozen=True)
class Run:
    """One (family, p, q, h, dt) run; ``check`` marks a dt/2 check.

    ``drift``, ``scheme`` and ``backend`` default to the article configuration (raw drift,
    IMEX-ARK3, device); other values tag the run name (``variant``).
    """

    family: str
    order: int
    offset: int
    mesh_size: float
    dt: float
    check: bool = False
    drift: str = "raw"
    scheme: str = "imex-ark3"
    backend: str = "device"
    steps: int | None = None
    repeat: int | None = None
    device_profile: str = "bsr"
    solver_rtol: float | None = None
    exact_drift_order_offset: int = 2
    exact_drift_mode: str = "pointwise"

    @property
    def poisson_order(self) -> int:
        return self.order + self.offset

    @property
    def final_time(self) -> float:
        """``FINAL_TIME``, or ``steps`` time steps for a timing run."""
        return FINAL_TIME if self.steps is None else self.steps * self.dt

    @property
    def drift_tag(self) -> str:
        """Drift in the run name: ``exact`` for the pointwise exact drift, ``exactp<s>`` for its DG(p+s) projection."""
        if self.drift != "exact":
            return self.drift
        return "exact" if self.exact_drift_mode == "pointwise" else f"exactp{self.exact_drift_order_offset}"

    @property
    def variant(self) -> str:
        """Run-name tag of a non-article configuration, e.g. ``_compatible_si-bdf3_host`` or ``_steps40_rep2``."""
        drift = self.drift_tag
        return ("".join(f"_{value}" for value, default in
                        ((drift, "raw"), (self.scheme, "imex-ark3"), (self.backend, "device"),
                         (self.device_profile, "bsr"))
                        if value != default) + ("" if self.steps is None else f"_steps{self.steps}")
                + ("" if self.repeat is None else f"_rep{self.repeat}")
                + ("" if self.solver_rtol is None else f"_rtol{self.solver_rtol:g}"))

    @property
    def name(self) -> str:
        return (f"{self.family}_p{self.order}_q{self.poisson_order}{self.variant}_h{self.mesh_size:g}"
                + ("_dthalf" if self.check else ""))


def case_params(family: str) -> dict[str, float]:
    """``rho_helm_wave`` parameters of the article's k = 8 pi wave in one family."""
    return {"kx": K8, "ky": FAMILIES[family]["ky"], "U": 1.0, "A": AMPLITUDE}


def plan(families, orders, offsets, mesh_sizes, dt_table: dict[str, float], checks: bool,
         **variant) -> list[Run]:
    """Runs in family, p, q order, coarse to fine, each ladder followed by its dt/2 check.

    ``variant``: ``drift``, ``scheme``, ``backend`` of every run (default: the article configuration).
    """
    runs = []
    for family in families:
        ladder = LADDERS[FAMILIES[family]["mesh_family"]]
        for order in orders:
            sizes = tuple(mesh_sizes) if mesh_sizes else ladder.get(order)
            if not sizes:
                raise ValueError(f"no {family} ladder for p={order}; pass --mesh-sizes")
            for offset in offsets:
                for size in sorted(sizes, reverse=True):
                    key = f"{order},{size:g}"
                    if key not in dt_table:
                        raise ValueError(f"dt table has no entry {key!r}")
                    runs.append(Run(family, order, offset, size, float(dt_table[key]), **variant))
                if checks:
                    finest = min(sizes)
                    runs.append(Run(family, order, offset, finest, 0.5 * float(dt_table[f"{order},{finest:g}"]),
                                    check=True, **variant))
    return runs


def _result_path(output: Path, run: Run) -> Path:
    return output / run.family / f"{run.name}.csv"


def _worker(spec: dict) -> None:
    """Run one case in this process (called through ``--worker``)."""
    from scripts.guiding_center.benchmarks.run_guiding_center_temporal_convergence import run_spatial_convergence

    run = Run(**spec["run"])
    run_spatial_convergence(
        scheme=run.scheme, final_time=run.final_time, orders=(run.order,), mesh_sizes=(run.mesh_size,),
        dt_for={(run.order, run.mesh_size): run.dt}, default_dt=run.dt,
        output_dir=Path(spec["output_dir"]) / run.family, prefix=run.name, verbosity=int(spec["verbosity"]),
        case_params=case_params(run.family), poisson_tau=1.0, backend=run.backend, device_profile=run.device_profile,
        mesh_family=FAMILIES[run.family]["mesh_family"], poisson_order_offset=run.offset,
        electric_field=run.drift, diagnostics_every=int(spec["log_every"]), solver_rtol=run.solver_rtol,
        exact_drift_order_offset=run.exact_drift_order_offset if run.drift == "exact" else None,
        exact_drift_mode=run.exact_drift_mode if run.drift == "exact" else None,
    )


@contextmanager
def _gpu_lock(path: Path | None, emit):
    """Hold the shared GPU campaign lock (``flock``-compatible) for one run."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit(f"[poisson-degree] waiting for the GPU lock {path}")
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _nvidia_smi(query: str) -> list[list[str]]:
    """Rows of one ``nvidia-smi --query-...`` call without header or units; empty if unavailable."""
    try:
        text = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True,
                              text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[value.strip() for value in line.split(",")] for line in text.splitlines() if line.strip()]


def _wait_for_gpu_memory(min_free_mib: int, emit, poll: float = 30.0) -> None:
    """Block until device 0 has ``min_free_mib`` MiB free, so a run never starts on a nearly full GPU."""
    since = None
    while min_free_mib > 0:
        rows = _nvidia_smi("--query-gpu=memory.free")
        if not rows:
            emit("[poisson-degree] nvidia-smi unavailable: free GPU memory not checked")
            return
        free = int(float(rows[0][0]))
        if free >= min_free_mib:
            if since is not None:
                emit(f"[poisson-degree] {free} MiB free after {time.perf_counter() - since:.0f}s")
            return
        if since is None:
            since = time.perf_counter()
            emit(f"[poisson-degree] waiting for {min_free_mib} MiB of free GPU memory ({free} MiB free)")
        time.sleep(poll)


def _gpu_lock_free(path: Path) -> bool:
    """Probe the GPU campaign lock without holding it: ``False`` while a timing campaign holds it."""
    if not path.exists():
        return True
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def _wait_while_gpu_lock_held(path: Path, emit, poll: float = 30.0, recheck: float = 10.0,
                              grace: float = 60.0) -> None:
    """Accuracy runs on the GPU: start only while no timing campaign holds the lock (never taken here).

    The lock must be free on two probes ``recheck`` seconds apart, and for ``grace`` seconds after a
    holder releases it, so a run does not slip in between two runs of an exclusive campaign.
    """
    since = None
    while True:
        if _gpu_lock_free(path):
            time.sleep(recheck if since is None else grace)
            if _gpu_lock_free(path):
                if since is not None:
                    emit(f"[poisson-degree] {path} free after {time.perf_counter() - since:.0f}s")
                return
        if since is None:
            since = time.perf_counter()
            emit(f"[poisson-degree] waiting: {path} is held by an exclusive GPU campaign")
        time.sleep(poll)


def _wait_for_exclusive_gpu(emit, poll: float = 30.0) -> None:
    """Block until no compute process runs on the GPU (timing runs; the lock alone does not stop other jobs)."""
    since = None
    while True:
        busy = [row[0] for row in _nvidia_smi("--query-compute-apps=pid") if row]
        if not busy:
            if since is not None:
                emit(f"[poisson-degree] GPU free after {time.perf_counter() - since:.0f}s")
            return
        if since is None:
            since = time.perf_counter()
            emit(f"[poisson-degree] waiting for an idle GPU (compute processes {', '.join(busy)})")
        time.sleep(poll)


class _PeakGpuMemory(threading.Thread):
    """Sample one process's device memory (nvidia-smi) every ``interval`` seconds; ``peak`` in MiB.

    ``others`` collects the pids of the other compute processes seen on the GPU meanwhile, and
    ``clocks``/``temperatures`` the SM clock (MHz) and temperature (C) samples (timing runs).
    """

    def __init__(self, pid: int, interval: float = 5.0):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak, self.done = str(pid), interval, 0, threading.Event()
        self.others: set[str] = set()
        self.clocks: list[float] = []
        self.temperatures: list[float] = []

    def run(self) -> None:
        while not self.done.wait(self.interval):
            for row in _nvidia_smi("--query-compute-apps=pid,used_memory"):
                if len(row) == 2 and row[0] == self.pid and row[1].isdigit():
                    self.peak = max(self.peak, int(row[1]))
                elif row and row[0] != self.pid:
                    self.others.add(row[0])
            for row in _nvidia_smi("--query-gpu=clocks.sm,temperature.gpu"):
                try:
                    self.clocks.append(float(row[0]))
                    self.temperatures.append(float(row[1]))
                except (IndexError, ValueError):
                    pass

    def stop(self) -> int:
        self.done.set()
        self.join()
        return self.peak


def _execute(run: Run, *, output: Path, verbosity: int, log_every: int, lock: Path | None, min_free_mib: int,
             emit, exclusive: bool = False, probe_lock: Path | None = None) -> int:
    """Run one worker process, streaming its lines (with an ETA on step summaries) to every log.

    ``exclusive`` (timing runs): after taking the lock, wait until no compute process runs on the
    GPU, then record in ``<log>.gpu.json`` whether another one appeared during the run.
    ``probe_lock`` (GPU accuracy runs): wait while an exclusive campaign holds that lock, without taking it.
    """
    spec = {"run": asdict(run), "output_dir": str(output), "verbosity": verbosity, "log_every": log_every}
    command = [sys.executable, "-u", "-m", "scripts.guiding_center.benchmarks.run_poisson_degree_study",
               "--worker", json.dumps(spec)]
    log_path = output / run.family / "logs" / f"{run.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _gpu_lock(lock, emit), log_path.open("w", buffering=1, encoding="utf-8") as log:
        def tee(line: str) -> None:
            emit(line)
            log.write(line + "\n")

        if exclusive:
            _wait_for_exclusive_gpu(emit)
        elif probe_lock is not None:
            _wait_while_gpu_lock_held(probe_lock, emit)
        _wait_for_gpu_memory(min_free_mib, emit)
        started = time.perf_counter()
        first = None  # (time, step) of the first step summary: the ETA excludes setup and JIT
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        memory = _PeakGpuMemory(process.pid)
        memory.start()
        for line in process.stdout:
            line = line.rstrip("\n")
            match = STEP_LINE.search(line)
            if match:
                step, total, now = int(match.group(1)), int(match.group(2)), time.perf_counter()
                if first is None or step <= first[1]:
                    first = (now, step)
                    line += f" elapsed={now - started:.0f}s"
                else:
                    rate = (now - first[0]) / (step - first[1])
                    line += f" elapsed={now - started:.0f}s eta={rate * (total - step):.0f}s"
            tee(line)
        status = process.wait()
        peak = memory.stop()
        tee(f"[poisson-degree] {run.name}: exit {status} after {time.perf_counter() - started:.0f}s, "
            f"peak GPU memory {peak} MiB")
        if exclusive:
            others = sorted(memory.others)
            if others:
                tee(f"[poisson-degree] WARNING {run.name}: other GPU processes during the run ({', '.join(others)}); "
                    "its timings are not exclusive")
            clocks, temperatures = sorted(memory.clocks), memory.temperatures
            log_path.with_suffix(".gpu.json").write_text(json.dumps(
                {"exclusive": not others, "other_pids": others, "peak_gpu_mib": peak,
                 "sm_clock_mhz_median": clocks[len(clocks) // 2] if clocks else None,
                 "sm_clock_mhz_min": clocks[0] if clocks else None,
                 "temperature_c_max": max(temperatures) if temperatures else None}) + "\n")
    return status


def _number(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _read_rows(path: Path, **extra) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{**{key: _number(value) for key, value in row.items() if value != ""}, **extra}
                for row in csv.DictReader(handle)]


def _variant_of(stem: str, family: str) -> str:
    """Variant tag of a run CSV stem (``Run.variant``; empty for the article configuration)."""
    match = re.match(rf"{re.escape(family)}_p\d+_q\d+(.*?)_h[^_]+(?:_dthalf)?$", stem)
    return match.group(1) if match else ""


def _rates(rows: list[dict]) -> None:
    """Orders against h = (|Omega|/N_K)^(1/2) between consecutive meshes of one (family, source, variant, p, q)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["source"] != "dthalf":
            groups.setdefault((row["family"], row["source"], row.get("variant", ""), int(row["order"]),
                               str(row["poisson_order"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: float(row["triangles"]))
        for coarse, fine in zip([None, *group[:-1]], group):
            for key in ("rho_l2_interior_error", "rho_l2_error", "phi_l2_error"):
                rate = key.replace("_error", "_rate")
                if coarse is None or not isinstance(coarse.get(key), float) or not isinstance(fine.get(key), float):
                    fine[rate] = None
                    continue
                ratio = math.sqrt(float(fine["triangles"]) / float(coarse["triangles"]))
                fine[rate] = math.log(coarse[key] / fine[key]) / math.log(ratio)


def combine(output: Path, baseline_dir: Path, exact_dir: Path, families, orders) -> Path:
    """Merge raised-q, baseline (q = p) and exact-drift rows; print orders and the dt/2 checks."""
    rows = []
    for family in families:
        settings = FAMILIES[family]
        for path in sorted((output / family).glob(f"{family}_p*_q*_h*.csv")):
            variant = _variant_of(path.stem, family)
            degrees = re.search(r"_p(\d+)_q(\d+)", path.stem)
            # The study's own article-configuration q = p runs are the baseline: all of it without an article
            # baseline, the meshes beyond the article ladder (ladder extension) otherwise.
            own_baseline = variant == "" and degrees is not None and degrees.group(1) == degrees.group(2)
            source = "dthalf" if path.stem.endswith("_dthalf") else "baseline" if own_baseline else "raised"
            rows += _read_rows(path, family=family, source=source, variant=variant)
        present = {int(row["order"]) for row in rows if row["family"] == family} or set(orders)
        for order in sorted(present):
            if settings["baseline"] is None:
                break
            path = baseline_dir / settings["baseline"].format(p=order)
            if path.exists():
                rows += _read_rows(path, family=family, source="baseline")
    for family in families:
        path = exact_dir / f"{family}_exact_drift.csv"
        if path.exists():
            rows += _read_rows(path, family=family, source="exact", poisson_order="exact")
    if not rows:
        raise SystemExit(f"no rows under {output} or {baseline_dir}")
    for row in rows:
        row.setdefault("poisson_order", row.get("order"))
        row.setdefault("variant", "")
        if isinstance(row["poisson_order"], float):
            row["poisson_order"] = int(row["poisson_order"])
    _rates(rows)
    source_rank = {"baseline": 0, "raised": 1, "exact": 2, "dthalf": 3}
    rows.sort(key=lambda row: (list(FAMILIES).index(row["family"]), int(row["order"]), source_rank[row["source"]],
                               row["variant"], str(row["poisson_order"]), float(row["triangles"])))
    merged = output / "poisson_degree.csv"
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, spec):
        return "-" if not isinstance(value, (int, float)) else format(value, spec)

    print(f"\n{'family':<11}{'p':>2}{'q':>6} {'variant':<26}{'K':>9}{'dt':>10}  {'rho int':>10}{'order':>7}  "
          f"{'rho L2':>10}{'order':>7}  {'phi L2':>10}{'order':>7}")
    for row in rows:
        if row["source"] == "dthalf":
            continue
        variant = row["variant"].lstrip("_") or ("article" if row["source"] != "exact" else "")
        print(f"{row['family']:<11}{int(row['order']):>2}{str(row['poisson_order']):>6} {variant:<26}"
              f"{int(row['triangles']):>9}"
              f"{fmt(row.get('dt'), '.3e'):>10}  {fmt(row.get('rho_l2_interior_error'), '.3e'):>10}"
              f"{fmt(row.get('rho_l2_interior_rate'), '.2f'):>7}  {fmt(row.get('rho_l2_error'), '.3e'):>10}"
              f"{fmt(row.get('rho_l2_rate'), '.2f'):>7}  {fmt(row.get('phi_l2_error'), '.3e'):>10}"
              f"{fmt(row.get('phi_l2_rate'), '.2f'):>7}")
    full = {(row["family"], int(row["order"]), str(row["poisson_order"]), row["variant"], int(row["triangles"])): row
            for row in rows if row["source"] in ("raised", "baseline")}
    for row in rows:
        if row["source"] != "dthalf":
            continue
        match = full.get((row["family"], int(row["order"]), str(row["poisson_order"]), row["variant"],
                          int(row["triangles"])))
        if match is None:
            continue
        changes = "  ".join(f"{key}: {abs(row[key] / match[key] - 1.0):.2%}"
                            for key in ("rho_l2_interior_error", "rho_l2_error", "phi_l2_error")
                            if isinstance(row.get(key), float) and isinstance(match.get(key), float))
        print(f"dt/2 check {row['family']} p={int(row['order'])} q={row['poisson_order']}{row['variant']} "
              f"K={int(row['triangles'])}: relative change {changes}")
    print(f"CSV: {merged}")
    return merged


def lattice(output: Path, sizes, family: str = "gmsh") -> Path:
    """Edge lattice of a Gmsh family's meshes in ``LATTICE_BOX`` -> ``<output>/<family>_lattice.json`` (no simulation)."""
    from hybridge.core.mesh import edge_lattice_orientation, gmsh_rectangle_mesh
    from scripts.guiding_center.benchmarks.run_guiding_center_temporal_convergence import mesh_family_gmsh_algorithm

    algorithm = mesh_family_gmsh_algorithm(FAMILIES[family]["mesh_family"])

    records = []
    for size in sorted(set(sizes), reverse=True):
        mesh = gmsh_rectangle_mesh(size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), algorithm=algorithm,
                                   num_threads=os.cpu_count())
        result = edge_lattice_orientation(mesh, box=LATTICE_BOX)
        records.append({"mesh_size": size, "triangles": int(mesh.triangles.shape[0]), "lattice": result})
        print(f"[lattice] h={size:<8g} N_K={records[-1]['triangles']:>8d} coherence={result['coherence']:.4f} "
              f"angle={result['angle']:6.2f} deg equilateral={result['equilateral']:.4f}", flush=True)
    path = output / f"{family}_lattice.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"family": family, "box": LATTICE_BOX, "meshes": records}, indent=1) + "\n")
    print(f"JSON: {path}")
    return path


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", default="misaligned", help=f"Comma list of {', '.join(FAMILIES)}.")
    parser.add_argument("--orders", default="2,3,4", help="Density degrees p.")
    parser.add_argument("--offsets", default="1,2",
                        help="Poisson degree offsets s (q = p + s); s = 0 only with a non-article --drift, "
                             "--scheme or --backend, or in another --output-dir (in the article directory the "
                             "article runs are the s = 0 baseline; the unstructured family has none).")
    parser.add_argument("--drift", choices=("raw", "compatible", "compatible-primal", "exact"), default="raw",
                        help="raw LDG-H flux; the compatible drift V2 (Oswald average of the p+1 postprocessed "
                             "potential; the raw-CUDA Poisson postprocesses on the device); compatible-primal "
                             "(Oswald average of phi_h itself, drift of degree p-1: the averaging without the "
                             "postprocess, tag _compatible-primal); or the exact drift at every solve time "
                             "(reference; --exact-drift-mode). Every drift but raw needs offset 0.")
    parser.add_argument("--scheme", choices=("imex-ark3", "si-bdf3", "si-bdf2"), default="imex-ark3",
                        help="Time scheme (the host backend has no IMEX-ARK3 path).")
    parser.add_argument("--backend", choices=("device", "host"), default="device",
                        help="device: raw-CUDA + AMGX (profile bsr); host: Numba + PyPardiso on every core.")
    parser.add_argument("--device-profile", default="bsr",
                        help="Device kernels/solvers of run_guiding_center_temporal_convergence (DEVICE_PROFILES); "
                             "bsr = article profile, csr for p >= 8 (the face-BSR transport DILU has blocks up to 8).")
    parser.add_argument("--steps", type=int, default=None,
                        help="Timing runs: stop each run after this many time steps (final time steps*dt; tags the "
                             "run name), enough for the median step time.")
    parser.add_argument("--ignore-gpu-lock", action="store_true",
                        help="GPU runs without --gpu-lock normally wait while an exclusive campaign holds the GPU "
                             "campaign lock (probe only, never taken); this disables the wait.")
    parser.add_argument("--exact-drift-mode", choices=("pointwise", "projected"), default="pointwise",
                        help="--drift exact: the exact drift at the transport's quadrature points (pointwise, tag "
                             "_exact) or the L2 projection of the exact flux onto DG(p+s) (projected, tag "
                             "_exactp<s>).")
    parser.add_argument("--exact-drift-order-offset", type=int, default=2,
                        help="--drift exact --exact-drift-mode projected: degree offset s of the projected exact "
                             "flux, DG(p+s) (default 2; tags the run name, e.g. _exactp3).")
    parser.add_argument("--solver-rtol", type=float, default=None,
                        help="Relative tolerance of the AMGX Poisson and transport solves (default 1e-11; absolute "
                             "tolerance a tenth of it); tags the run name, so it needs its own --output-dir.")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Timing runs: repetition number, tags the run name (_rep<N>) so a run can be repeated.")
    parser.add_argument("--exclusive-gpu", action="store_true",
                        help="Timing runs: hold the GPU lock (--gpu-lock, default the campaign lock), wait until no "
                             "other compute process runs on the GPU and record any that appears during a run.")
    parser.add_argument("--mesh-sizes", default=None, help="Comma list overriding every family's ladder.")
    parser.add_argument("--checks", action="store_true",
                        help="Add a dt/2 run on the finest mesh of every (family, p, q).")
    parser.add_argument("--dt-table", type=Path, default=DEFAULT_BASELINE / "dt_table.json",
                        help='JSON {"p,h": dt} of the article runs.')
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE,
                        help="Article runs supplying the q = p rows.")
    parser.add_argument("--exact-drift-dir", type=Path, default=DEFAULT_EXACT,
                        help="Directory of the exact-drift CSVs <family>_exact_drift.csv (q = infinity).")
    parser.add_argument("-v", "--verbosity", type=int, choices=(0, 1, 2, 3), default=1,
                        help="0 quiet, 1 step summaries, 2 solver phases, 3 every timing.")
    parser.add_argument("--log-every", type=int, default=10, help="Steps between step summaries (diagnostics).")
    parser.add_argument("--gpu-lock", type=Path, default=None,
                        help=f"flock file held during each run, for timing studies only (e.g. {GPU_LOCK}).")
    parser.add_argument("--min-free-gpu-mib", type=int, default=8192,
                        help="Free device memory (MiB) to wait for before each run; 0 disables the check.")
    parser.add_argument("--rerun", action="store_true", help="Rerun runs whose CSV exists.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List the runs and their status only.")
    parser.add_argument("--combine-only", action="store_true", help="Only merge and report existing rows.")
    parser.add_argument("--lattice", action="store_true",
                        help="Only measure the edge lattice of the meshes of the Gmsh families in --families "
                             "(default gmsh; their ladders, plus EXTRA_GMSH_SIZES for gmsh, or --mesh-sizes) -> "
                             "<output>/<family>_lattice.json; no simulation.")
    parser.add_argument("--worker", default=None, help=SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        _worker(json.loads(args.worker))
        return
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        parser.error(f"unknown families {unknown}; choose from {sorted(FAMILIES)}")
    if args.lattice:
        for family in (families if args.families != parser.get_default("families") else ["gmsh"]):
            mesh_family = FAMILIES[family]["mesh_family"]
            if mesh_family == "structured":
                parser.error(f"--lattice measures Gmsh meshes; {family} is structured")
            sizes = ([float(value) for value in args.mesh_sizes.split(",")] if args.mesh_sizes else
                     [size for ladder in LADDERS[mesh_family].values() for size in ladder]
                     + (list(EXTRA_GMSH_SIZES) if family == "gmsh" else []))
            lattice(args.output_dir, sizes, family)
        return
    orders = [int(value) for value in args.orders.split(",")]
    offsets = [int(value) for value in args.offsets.split(",")]
    if args.drift != "raw" and any(offset != 0 for offset in offsets):
        parser.error(f"--drift {args.drift} keeps the Poisson degree of the density: use --offsets 0")
    if args.exact_drift_mode != "pointwise" and args.drift != "exact":
        parser.error("--exact-drift-mode applies to --drift exact only")
    if args.exact_drift_order_offset != 2 and (args.drift != "exact" or args.exact_drift_mode != "projected"):
        parser.error("--exact-drift-order-offset applies to --drift exact --exact-drift-mode projected only")
    if args.exact_drift_order_offset < 0:
        parser.error("--exact-drift-order-offset must be nonnegative")
    if args.steps is not None and args.steps < 3:
        parser.error("--steps must be at least 3 (the median step time skips the first two steps)")
    variant = {"drift": args.drift, "scheme": args.scheme, "backend": args.backend, "steps": args.steps,
               "repeat": args.repeat, "device_profile": args.device_profile, "solver_rtol": args.solver_rtol,
               "exact_drift_order_offset": args.exact_drift_order_offset, "exact_drift_mode": args.exact_drift_mode}
    article = Run("gmsh", 1, 1, 1.0, 1.0, **variant).variant == ""
    output = args.output_dir
    default_output = output.resolve() == DEFAULT_OUTPUT.resolve()
    mesh_sizes = [float(value) for value in args.mesh_sizes.split(",")] if args.mesh_sizes else None
    # In the article directory offset 0 only extends the article ladders (meshes outside them).
    extension = mesh_sizes is not None and all(
        size not in LADDERS[FAMILIES[family]["mesh_family"]].get(order, ())
        for family in families if FAMILIES[family]["baseline"] is not None for order in orders for size in mesh_sizes)
    if any(offset < 0 for offset in offsets) or (article and 0 in offsets and default_output and not extension):
        parser.error("offsets must be positive in the article directory (offset 0 is the article baseline) unless "
                     "--mesh-sizes extends the article ladders; rerun offset 0 with a non-article configuration "
                     "or in another --output-dir")
    if args.backend == "host" and args.scheme == "imex-ark3":
        parser.error("the host backend has no IMEX-ARK3 path; use --scheme si-bdf3")
    if not article and default_output:
        parser.error("non-article configurations need their own --output-dir: the default directory holds "
                     "the article's poisson_degree.csv")
    if args.combine_only:
        combine(output, args.baseline_dir, args.exact_drift_dir, families, orders)
        return
    runs = plan(families, orders, offsets, mesh_sizes, json.loads(args.dt_table.read_text()), args.checks,
                **variant)
    pending = [run for run in runs if args.rerun or not _result_path(output, run).exists()]
    if args.dry_run:
        for run in runs:
            status = "pending" if run in pending else "done"
            print(f"{run.name:<56} dt={run.dt:.4e} steps={math.ceil(run.final_time / run.dt - 1e-9):5d} {status}")
        print(f"{len(pending)} of {len(runs)} runs pending; output {output}")
        return

    output.mkdir(parents=True, exist_ok=True)
    lock = args.gpu_lock if args.gpu_lock is not None or not args.exclusive_gpu else GPU_LOCK
    # GPU runs without the lock wait while an exclusive (timing) campaign holds it; host runs never wait.
    probe_lock = (None if lock is not None or args.backend == "host" or args.ignore_gpu_lock else GPU_LOCK)
    failures = []
    with (output / "campaign.log").open("a", buffering=1, encoding="utf-8") as campaign:
        def emit(line: str) -> None:
            print(line, flush=True)
            campaign.write(line + "\n")

        emit(f"[poisson-degree] {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(pending)} of {len(runs)} runs "
             f"pending (families={families}, p={orders}, s={offsets}, checks={args.checks}, {variant}, "
             f"v={args.verbosity})")
        for index, run in enumerate(pending, 1):
            emit(f"[poisson-degree] run {index}/{len(pending)}: {run.name} dt={run.dt:.4e}")
            status = _execute(run, output=output, verbosity=args.verbosity, log_every=args.log_every,
                              lock=lock, min_free_mib=args.min_free_gpu_mib, emit=emit,
                              exclusive=args.exclusive_gpu, probe_lock=probe_lock)
            if status != 0 or not _result_path(output, run).exists():
                failures.append({"run": run.name, "exit_status": status})
                (output / "failures.json").write_text(json.dumps(failures, indent=1) + "\n")
                if args.stop_on_failure:
                    break
        emit(f"[poisson-degree] {len(failures)} failed run(s)" if failures else "[poisson-degree] all runs done")
    combine(output, args.baseline_dir, args.exact_drift_dir, families, orders)


if __name__ == "__main__":
    main()
