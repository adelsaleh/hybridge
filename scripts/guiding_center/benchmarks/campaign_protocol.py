"""Protocol shared by the detached GPU performance campaigns (D8-S smoke, D8): one process per run.

A campaign launcher (a script in its study directory, started with ``setsid nohup``) builds the run
specifications and the driver command lines; this module supplies everything else of the protocol of
PLAN.md D8 "Performance protocol":

- the campaign-wide GPU lock ``hybridge/.cache/gpu-campaign.lock`` (``hold_lock``), held for the whole
  campaign so that other sessions' accuracy runs, which probe it, stay off the device;
- an idle GPU before each run (``wait_for_idle_gpu``: no compute process in nvidia-smi, polled every 30 s);
- host process groups listed in ``<study>/pause_pgids.json`` paused (SIGSTOP) during each run and resumed
  after it and on exit;
- nvidia-smi every 2 s during a run (``Sampler`` → ``<runs>/gpu/<run_id>_attempt<n>.csv``: compute
  processes with their memory, utilization, memory, clocks, temperature, power);
- contamination: a run during which a foreign PID computes on the device with more than 1 GB, or for
  longer than 30 s and 5 % of the run, is recorded as ``contaminated``, its outputs moved to
  ``<study>/contaminated/`` and the run queued once more (``finished`` returns False until
  ``max_attempts`` attempts); smaller and shorter foreign contexts (RustDesk's hourly encoder probe,
  336 MiB for seconds) are tolerated, logged and recorded in ``foreign_processes`` with their names;
- host load and the top CPU consumers at start and end, peak device memory of the run's processes and of
  the device, temperature, the run's wall time, the exit status (``ok`` / ``failed`` / ``oom`` /
  ``timeout`` / ``contaminated``), appended to ``<study>/runs.jsonl`` (``execute``);
- provenance (``provenance``: hybridge commit, sha256 of the tracked diff and of the untracked sources
  that are copied into ``<study>/source/``, GPU, driver, CUDA, Python, launcher arguments);
- resuming: ``finished(run_id)`` from the records, so a restarted launcher skips completed runs.

Runs are confined to the cores in ``cpu_cores`` with ``taskset`` (another session's MPI study binds its
ranks to cores 0-11, 2026-10-16).

Accuracy campaigns use ``Campaign(..., exclusive=False, min_free_mib=...)``: they never take the lock,
wait before each run until no performance campaign holds it and the device has ``min_free_mib`` free
(``wait_for_free_device``), pause no host groups, and record foreign GPU processes without a
contamination verdict.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HYBRIDGE = Path(__file__).resolve().parents[3]
PYTHON = HYBRIDGE / ".venv" / "bin" / "python"
LOCK = HYBRIDGE / ".cache" / "gpu-campaign.lock"
OOM_TOKENS = ("OutOfMemory", "out of memory", "cudaErrorMemoryAllocation")
# A foreign compute process contaminates a run only above these: device memory in MiB, or lifetime in
# seconds and as a fraction of the run's wall time (both must be exceeded for the lifetime criterion).
MINOR_INTRUDER_MIB = 1024.0
MINOR_INTRUDER_SECONDS = 30.0
MINOR_INTRUDER_FRACTION = 0.05


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


# Per-step timing keys of the runner's ``*_timings.jsonl`` summarized in the driver log after each run
# (medians over the steps after the first two, in ms; iteration counts as medians).
TIMING_KEYS = (("step_advance_wall_time", "step"), ("poisson_time_total", "poisson"),
               ("poisson_krylov_time", "poisson_krylov"), ("poisson_time_postprocessing", "postprocess"),
               ("transport_time_assembly", "transport_assembly"), ("transport_time_solve", "transport_solve"),
               ("enstrophy_budget_time", "enstrophy"))
ITERATION_KEYS = (("poisson_solver_iterations", "poisson_its"), ("transport_solver_iterations", "transport_its"))


def timing_summary(runs: Path, run_id: str, skip: int = 2, pattern: str | None = None) -> dict:
    """Medians of the per-step timings of one run (``<runs>/*_<run_id>_seed_timings.jsonl``).

    Steps 1..``skip`` are excluded (JIT, cold solves); the first step's Poisson Krylov time is reported
    separately as ``cold_poisson_krylov``. Returns ``{}`` when the run wrote no timings.
    """
    path = next(runs.glob(pattern or f"*_{run_id}_seed_timings.jsonl"), None)
    if path is None:
        return {}
    rows = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    steps = [row for row in rows if isinstance(row.get("step"), (int, float))]
    warm = [row for row in steps if row["step"] > skip] or steps
    out = {"steps": len(steps)}
    for key, name in TIMING_KEYS:
        value = _median(float(row[key]) for row in warm if isinstance(row.get(key), (int, float)))
        if value is not None:
            out[name + "_ms"] = 1000.0 * value
    for key, name in ITERATION_KEYS:
        value = _median(float(row[key]) for row in warm if isinstance(row.get(key), (int, float)))
        if value is not None:
            out[name] = value
    # The cold solve is the initial Poisson solve (row ``step == "initial"``), else the first step's.
    first = next((row for row in rows if row.get("step") == "initial"), None) or (steps[0] if steps else None)
    if first is not None and isinstance(first.get("poisson_krylov_time"), (int, float)):
        out["cold_poisson_krylov_ms"] = 1000.0 * first["poisson_krylov_time"]
    return out


def format_timings(summary: dict) -> str:
    """One line: ``step 302 ms | poisson 61 (krylov 19, 1.5 it) | transport asm 120 + solve 90 (21 it) | ...``."""
    if not summary:
        return "no per-step timings"
    def ms(name):
        value = summary.get(name + "_ms")
        return "-" if value is None else f"{value:.0f}"
    def its(name):
        value = summary.get(name)
        return "-" if value is None else f"{value:g}"
    parts = [f"step {ms('step')} ms over {summary.get('steps', 0)} steps",
             f"poisson {ms('poisson')} (krylov {ms('poisson_krylov')}, {its('poisson_its')} it, cold "
             f"{ms('cold_poisson_krylov')})",
             f"transport assembly {ms('transport_assembly')} + solve {ms('transport_solve')} "
             f"({its('transport_its')} it)"]
    if summary.get("postprocess_ms"):
        parts.append(f"postprocess {ms('postprocess')}")
    if summary.get("enstrophy_ms") is not None:
        parts.append(f"enstrophy budget {ms('enstrophy')} (outside the step)")
    return " | ".join(parts)


def compute_apps() -> list[tuple[int, int]]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    apps = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            apps.append((int(parts[0]), int(parts[1]) if parts[1].isdigit() else -1))
    return apps


def gpu_state() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,clocks.mem,"
                          "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return [part.strip() for part in out.strip().split(",")]


def wait_for_idle_gpu() -> float:
    start, logged = time.time(), False
    while apps := compute_apps():
        if not logged:
            log(f"waiting for an idle GPU: {apps}")
            logged = True
        time.sleep(30)
    return time.time() - start


def lock_held() -> bool:
    """True while another process holds the performance-campaign GPU lock (probe only, never kept)."""
    if not LOCK.exists():
        return False
    with LOCK.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def free_device_mib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    used, total = (float(part) for part in out.split(","))
    return total - used


def wait_for_free_device(min_free_mib: float, poll: float = 60.0) -> float:
    """Accuracy runs: wait while a performance campaign holds the lock or free memory is short."""
    start, logged = time.time(), None
    while True:
        held, free = lock_held(), free_device_mib()
        if not held and free >= min_free_mib:
            return time.time() - start
        reason = "performance-campaign lock held" if held else f"{free:.0f} MiB free < {min_free_mib:.0f}"
        if reason != logged:
            log(f"waiting for the device: {reason}")
            logged = reason
        time.sleep(poll)


class Sampler(threading.Thread):
    """nvidia-smi every 2 s: compute processes and device state, into <run>.gpu.csv."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path, self.stop_event, self.rows = path, threading.Event(), []
        self.names: dict[int, str] = {}  # pid -> command name, looked up while the process still exists

    def run(self) -> None:
        while not self.stop_event.is_set():
            stamp = time.time()
            try:
                apps, state = compute_apps(), gpu_state()
            except Exception as error:  # keep sampling
                apps, state = [], [f"error {error}"]
            self.rows.append((stamp, apps, state))
            for pid, _ in apps:
                if pid not in self.names:
                    name = subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True,
                                          text=True).stdout.strip()
                    self.names[pid] = name or "?"
            self.stop_event.wait(max(0.0, 2.0 - (time.time() - stamp)))

    def finish(self) -> None:
        self.stop_event.set()
        self.join(timeout=30)
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", "pid", "used_mib", "utilization_gpu", "memory_used_mib", "clock_sm",
                             "clock_mem", "temperature", "power"])
            for stamp, apps, state in self.rows:
                for pid, used in apps or [("", "")]:
                    writer.writerow([f"{stamp:.3f}", pid, used, *state])


def descendants(pid: int) -> set[int]:
    out = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True, text=True).stdout.split("\n")[1:]
    children: dict[int, list[int]] = {}
    for line in out:
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, stack = {pid}, [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def host_load() -> dict:
    top = subprocess.run(["ps", "-eo", "pid,pgid,pcpu,comm", "--sort=-pcpu"], capture_output=True,
                         text=True).stdout.splitlines()[1:6]
    return dict(loadavg=os.getloadavg(), top=[line.strip() for line in top])


def paused_groups(study: Path) -> list[int]:
    path = study / "pause_pgids.json"
    try:
        return [int(value) for value in json.loads(path.read_text())] if path.exists() else []
    except Exception as error:
        log(f"ignoring unreadable {path.name}: {error}")
        return []


def signal_groups(groups, number) -> None:
    for group in groups:
        try:
            os.killpg(group, number)
            log(f"{'paused' if number == signal.SIGSTOP else 'resumed'} host process group {group}")
        except ProcessLookupError:
            pass


def _median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


class Campaign:
    """The protocol state of one campaign: its study directory, records and settings."""

    def __init__(self, study: Path, *, cpu_cores: str = "12-23", run_timeout: float = 1800.0, max_attempts: int = 2,
                 runs_dir: str = "runs", records_name: str = "runs.jsonl", exclusive: bool = True,
                 min_free_mib: float = 8192.0):
        self.study = Path(study)
        self.runs = self.study / runs_dir
        self.records_path = self.study / records_name
        self.cpu_cores, self.run_timeout, self.max_attempts = cpu_cores, float(run_timeout), int(max_attempts)
        self.current: subprocess.Popen | None = None
        self._lock = None
        self.exclusive, self.min_free_mib = bool(exclusive), float(min_free_mib)

    # records -------------------------------------------------------------------------------------
    def records(self) -> list[dict]:
        if not self.records_path.exists():
            return []
        return [json.loads(line) for line in self.records_path.read_text().splitlines() if line.strip()]

    def finished(self, run_id: str) -> bool:
        attempts = [row for row in self.records() if row["run_id"] == run_id]
        return any(row["status"] != "contaminated" for row in attempts) or len(attempts) >= self.max_attempts

    def append(self, row: dict) -> None:
        with self.records_path.open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")

    # lock and signals ----------------------------------------------------------------------------
    def hold_lock(self) -> None:
        """Take the campaign-wide GPU lock (blocking) and install SIGTERM/SIGINT handlers that stop the
        current run, resume the paused host groups and exit."""
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        self._lock = LOCK.open("a")
        log(f"waiting for {LOCK}")
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        log("GPU lock held for the whole campaign")

        def stop(number, frame):
            child = self.current
            if child is not None and child.poll() is None:
                child.terminate()
            signal_groups(paused_groups(self.study), signal.SIGCONT)
            log(f"stopped by signal {number}")
            sys.exit(1)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def release(self) -> None:
        signal_groups(paused_groups(self.study), signal.SIGCONT)
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    @staticmethod
    def wait_for(paths) -> None:
        for path in paths:
            if not Path(path).exists():
                log(f"waiting for {path}")
                while not Path(path).exists():
                    time.sleep(60)

    # one run -------------------------------------------------------------------------------------
    def execute(self, run: dict, cmd: list[str], level: dict, *, foreign_test: bool = False,
                output_glob: str | None = None, timings_glob: str | None = None) -> dict:
        """Run one driver command under the protocol and append its record. ``run`` must carry
        ``run_id``; ``level`` the mesh row (``triangles``, ``element_unknowns``, ``trace_unknowns``, ``hb``,
        ``h``), copied into the record. With ``foreign_test`` (first attempt only) a deliberate foreign
        GPU process starts 20 s into the run: the tooling must flag it and repeat the run.
        ``output_glob`` selects the run's outputs to move on contamination (default ``*_<run_id>_seed*``)."""
        attempt = 1 + sum(row["run_id"] == run["run_id"] for row in self.records())
        waited = wait_for_idle_gpu() if self.exclusive else wait_for_free_device(self.min_free_mib)
        groups = paused_groups(self.study) if self.exclusive else []
        signal_groups(groups, signal.SIGSTOP)
        load_start = host_load()
        (self.runs / "logs").mkdir(parents=True, exist_ok=True)
        (self.runs / "gpu").mkdir(parents=True, exist_ok=True)
        log_path = self.runs / "logs" / f"{run['run_id']}.log"
        sampler = Sampler(self.runs / "gpu" / f"{run['run_id']}_attempt{attempt}.csv")
        log(f"start {run['run_id']} (attempt {attempt}; {level.get('triangles')} triangles)")
        started = time.time()
        sampler.start()
        foreign, timer, exit_code, timed_out = None, None, None, False
        try:
            with log_path.open("w") as stream:
                stream.write(" ".join(cmd) + "\n")
                stream.flush()
                child = subprocess.Popen(["taskset", "-c", self.cpu_cores, *cmd], cwd=HYBRIDGE, stdout=stream,
                                         stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONPATH=str(HYBRIDGE)))
                self.current = child
                if foreign_test and attempt == 1:
                    def start_foreign():
                        nonlocal foreign
                        foreign = subprocess.Popen([str(PYTHON), "-c",
                                                    "import cupy, time; cupy.zeros(1); time.sleep(25)"])
                        log(f"deliberate foreign GPU process {foreign.pid} started")
                    timer = threading.Timer(20.0, start_foreign)
                    timer.start()
                own = {child.pid}
                while child.poll() is None:
                    own |= descendants(child.pid)
                    if time.time() - started > self.run_timeout:
                        timed_out = True
                        child.terminate()
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            child.kill()
                        break
                    time.sleep(1.0)
                exit_code = child.wait()
        finally:
            self.current = None
            if timer is not None:
                timer.cancel()
                timer.join()
            sampler.finish()
            if foreign is not None:
                foreign.wait(timeout=60)
            signal_groups(groups, signal.SIGCONT)
        wall = time.time() - started
        seen = {pid for _, apps, _ in sampler.rows for pid, _ in apps}
        # Foreign compute processes: a run is contaminated only by one that could have shared the device
        # for real (more than MINOR_INTRUDER_MIB, or present longer than MINOR_INTRUDER_SECONDS and
        # MINOR_INTRUDER_FRACTION of the run). Smaller, shorter ones are recorded and logged but tolerated:
        # RustDesk's hourly server restart (xx:46) opens a 336 MiB encoder context for a few seconds
        # (2026-10-17, three smoke runs redone for it).
        foreign = {}
        for stamp, apps, _ in sampler.rows:
            for pid, used in apps:
                if pid not in own:
                    entry = foreign.setdefault(pid, dict(pid=pid, name=sampler.names.get(pid, "?"), samples=0,
                                                         max_mib=0.0, first=stamp, last=stamp))
                    entry["samples"] += 1
                    entry["max_mib"] = max(entry["max_mib"], float(used or 0))
                    entry["last"] = stamp
        for entry in foreign.values():
            entry["seconds"] = round(entry["last"] - entry["first"] + 2.0, 1)
            entry["serious"] = (entry["max_mib"] > MINOR_INTRUDER_MIB
                                or entry["seconds"] > max(MINOR_INTRUDER_SECONDS, MINOR_INTRUDER_FRACTION * wall))
        intruders = sorted(pid for pid, entry in foreign.items() if entry["serious"]) if self.exclusive else []
        minor = sorted(pid for pid, entry in foreign.items() if not entry["serious"])
        if minor:
            log("tolerated foreign GPU process(es): " + ", ".join(
                f"{pid} {foreign[pid]['name']} {foreign[pid]['max_mib']:.0f} MiB {foreign[pid]['seconds']} s"
                for pid in minor))
        peak_process = max((used for _, apps, _ in sampler.rows for pid, used in apps if pid in own), default=None)
        peak_total = max((float(state[1]) for _, _, state in sampler.rows if len(state) > 1
                          and state[1].replace(".", "").isdigit()), default=None)
        # Device clock and temperature while the GPU computes (utilization above 50 %): a hot GPU clocks down.
        active = [state for _, _, state in sampler.rows if len(state) >= 6 and state[0].isdigit() and int(state[0]) > 50]
        sm_clock = _median(float(state[2]) for state in active if state[2].replace(".", "").isdigit())
        temperature = max((float(state[4]) for state in active if state[4].replace(".", "").isdigit()), default=None)
        text = log_path.read_text(errors="replace")
        status = "ok" if exit_code == 0 and "FAILED" not in text else "failed"
        if any(token in text for token in OOM_TOKENS):
            status = "oom"
        if timed_out:
            status = "timeout"
        if intruders:
            status = "contaminated"
            target = self.study / "contaminated" / f"{run['run_id']}_attempt{attempt}"
            target.mkdir(parents=True, exist_ok=True)
            for path in self.runs.glob(output_glob or f"*_{run['run_id']}_seed*"):
                shutil.move(str(path), target / path.name)
            shutil.copy(log_path, target / log_path.name)
        row = dict(run, status=status, attempt=attempt, exit_code=exit_code, wall_s=wall, gpu_wait_s=waited,
                   peak_process_mib=peak_process, peak_total_mib=peak_total, intruders=intruders,
                   foreign_processes=sorted(foreign.values(), key=lambda entry: entry["first"]),
                   sm_clock_mhz_active=sm_clock, temperature_max_c=temperature, active_samples=len(active),
                   samples=len(sampler.rows), load_start=load_start, load_end=host_load(), paused_groups=groups,
                   started=datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                   ended=datetime.now().isoformat(timespec="seconds"), triangles=level.get("triangles"),
                   element_unknowns=level.get("element_unknowns"), trace_unknowns=level.get("trace_unknowns"),
                   hb=level.get("hb"), h=level.get("h"), log=str(log_path))
        timings = timing_summary(self.runs, run["run_id"], pattern=timings_glob) if status != "contaminated" else {}
        row["timings"] = timings
        self.append(row)
        log(f"end {run['run_id']}: {status} in {wall:.0f} s, peak {peak_process} MiB"
            + (f", foreign GPU processes {intruders}" if intruders else ""))
        log(f"timings {run['run_id']}: {format_timings(timings)}")
        return row

    # provenance ----------------------------------------------------------------------------------
    def provenance(self, **extra) -> dict:
        """Append a provenance line (hybridge commit, sha256 of the tracked diff and the untracked sources,
        copies of them in <study>/source/, GPU, driver, CUDA, Python, argv, host load) to provenance.jsonl."""
        def git(*args):
            return subprocess.run(["git", "-C", str(HYBRIDGE), *args], capture_output=True, text=True).stdout

        source = self.study / "source"
        source.mkdir(exist_ok=True)
        diff = git("diff", "HEAD")
        (source / "hybridge_tracked.diff").write_text(diff)
        untracked = [line for line in git("ls-files", "--others", "--exclude-standard").splitlines()
                     if line.endswith((".py", ".json")) and line.startswith(("hybridge/", "scripts/", "configs/"))]
        digest = hashlib.sha256(diff.encode())
        for name in untracked:
            target = source / "untracked" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(HYBRIDGE / name, target)
            digest.update(name.encode() + (HYBRIDGE / name).read_bytes())
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.splitlines()
        info = dict(started=datetime.now().isoformat(timespec="seconds"), commit=git("rev-parse", "HEAD").strip(),
                    source_sha256=digest.hexdigest(), untracked=untracked, argv=sys.argv,
                    gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                    cuda=next((line.strip() for line in smi if "CUDA Version" in line), None),
                    python=sys.version, cpu_cores=self.cpu_cores, host_load=host_load(), **extra)
        with (self.study / "provenance.jsonl").open("a") as stream:
            stream.write(json.dumps(info) + "\n")
        return info
