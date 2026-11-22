"""Linux CPU sampling in a separate process, independent of the solver's GIL.

CPU equivalents measure scheduled CPU time / elapsed time, including spin waits.
Thread activity is resolved at the kernel clock-tick and sampling intervals; it
does not measure hardware execution-unit occupancy or useful arithmetic work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_stat(text):
    # comm may itself contain spaces and parentheses.
    fields = text.rsplit(")", 1)[1].split()
    return dict(state=fields[0], ticks=int(fields[11]) + int(fields[12]),
                last_cpu=int(fields[36]))


def snapshot(pid):
    root = Path(f"/proc/{pid}")
    process = parse_stat((root / "stat").read_text())
    threads = {}
    for entry in (root / "task").iterdir():
        try:
            threads[int(entry.name)] = parse_stat((entry / "stat").read_text())
        except (FileNotFoundError, ProcessLookupError):
            pass
    rss = 0
    for line in (root / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
            break
    return dict(time=time.monotonic(), ticks=process["ticks"], threads=threads, rss=rss)


def sample(pid, interval, events_path, csv_path, thread_path, stop_path, capacity):
    hz = os.sysconf("SC_CLK_TCK")
    origin = time.monotonic()
    phase = {"sequence": -1, "phase": "startup"}
    with open(events_path) as events, open(csv_path, "w") as output, open(thread_path, "w") as thread_output:
        writer = csv.DictWriter(output, fieldnames=["elapsed_seconds", "phase", "interval_seconds",
                               "cpu_seconds", "cpu_equivalent_cores", "allocated_cpu_percent",
                               "threads", "active_threads", "runnable_threads", "rss_gib"])
        thread_writer = csv.DictWriter(thread_output, fieldnames=["elapsed_seconds", "phase", "tid",
                                      "cpu_seconds", "cpu_percent", "state", "last_cpu"])
        writer.writeheader()
        thread_writer.writeheader()
        output.flush()
        thread_output.flush()

        def current_phase():
            nonlocal phase
            # Parent writes and flushes a single short JSON line per boundary.
            while True:
                position = events.tell()
                line = events.readline()
                if not line or not line.endswith("\n"):
                    events.seek(position)
                    break
                phase = json.loads(line)
            return phase

        previous = None
        previous_sequence = None
        while not stop_path.exists():
            started = time.monotonic()
            before = current_phase()
            try:
                now = snapshot(pid)
            except (FileNotFoundError, ProcessLookupError):
                break
            after = current_phase()
            if previous is not None:
                dt = now["time"] - previous["time"]
                stable = before["sequence"] == after["sequence"] == previous_sequence
                name = after["phase"] if stable else "boundary"
                cpu = max(0, now["ticks"] - previous["ticks"]) / hz
                active = 0
                for tid, stat in now["threads"].items():
                    old = previous["threads"].get(tid)
                    # A newly observed thread's counters start at zero.
                    delta = max(0, stat["ticks"] - (old["ticks"] if old else 0)) / hz
                    active += int(delta > 0)
                    thread_writer.writerow(dict(elapsed_seconds=now["time"] - origin, phase=name, tid=tid,
                                                cpu_seconds=delta, cpu_percent=100 * delta / dt,
                                                state=stat["state"], last_cpu=stat["last_cpu"]))
                writer.writerow(dict(elapsed_seconds=now["time"] - origin, phase=name, interval_seconds=dt,
                                     cpu_seconds=cpu, cpu_equivalent_cores=cpu / dt,
                                     allocated_cpu_percent=100 * cpu / dt / capacity,
                                     threads=len(now["threads"]), active_threads=active,
                                     runnable_threads=sum(s["state"] == "R" for s in now["threads"].values()),
                                     rss_gib=now["rss"] / 1024**3))
                output.flush()
                thread_output.flush()
            previous, previous_sequence = now, after["sequence"]
            time.sleep(max(0, interval - (time.monotonic() - started)))


class CpuProfile:
    def __init__(self, report_path, enabled=False, interval=0.2):
        self.enabled = enabled
        self.interval = interval
        self.phases = []
        self.sequence = 0
        self.stack = []
        self.process = None
        self.capacity = len(os.sched_getaffinity(0))
        self.paths = {key: Path(report_path).with_suffix(suffix) for key, suffix in {
            "samples_csv": ".cpu.csv", "threads_csv": ".cpu_threads.csv",
            "events_jsonl": ".cpu_events.jsonl", "sampler_log": ".cpu_sampler.log",
            "stop": ".cpu_stop"}.items()}
        if enabled:
            self.paths["events_jsonl"].parent.mkdir(parents=True, exist_ok=True)
            self.paths["stop"].unlink(missing_ok=True)
            self.events = self.paths["events_jsonl"].open("w")
            self.sampler_log = self.paths["sampler_log"].open("w")
            self.process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--pid", str(os.getpid()),
                "--interval", str(interval), "--events", str(self.paths["events_jsonl"]),
                "--output", str(self.paths["samples_csv"]), "--threads-output", str(self.paths["threads_csv"]),
                "--stop", str(self.paths["stop"]), "--capacity", str(self.capacity)],
                stdout=self.sampler_log, stderr=subprocess.STDOUT)

    def event(self, phase):
        self.sequence += 1
        self.events.write(json.dumps(dict(sequence=self.sequence, phase=phase,
                                         monotonic_seconds=time.monotonic())) + "\n")
        self.events.flush()

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.event(name)
        self.stack.append(name)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
            self.phases.append(dict(phase=name, wall_seconds=wall, cpu_seconds=cpu,
                                    average_cpu_equivalent_cores=cpu / wall,
                                    allocated_cpu_percent=100 * cpu / wall / self.capacity))
            self.stack.pop()
            self.event(self.stack[-1] if self.stack else "idle")

    def finish(self):
        if not self.enabled:
            return None
        self.event("finished")
        self.paths["stop"].touch()
        try:
            self.process.wait(timeout=max(5, 2 * self.interval))
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.events.close()
        self.sampler_log.close()
        self.paths["stop"].unlink(missing_ok=True)
        rows = []
        if self.paths["samples_csv"].exists():
            with self.paths["samples_csv"].open() as stream:
                rows = list(csv.DictReader(stream))
        for phase in self.phases:
            selected = [r for r in rows if r["phase"] == phase["phase"] or
                        r["phase"].startswith(phase["phase"] + ".")]
            phase["stable_samples"] = len(selected)
            if selected:
                phase.update(peak_sample_cpu_equivalent_cores=max(float(r["cpu_equivalent_cores"]) for r in selected),
                             peak_active_threads=max(int(r["active_threads"]) for r in selected),
                             peak_runnable_threads=max(int(r["runnable_threads"]) for r in selected))
        return dict(enabled=True, interval_seconds=self.interval, clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
                    capacity_logical_cpus=self.capacity, phases=self.phases, sampler_returncode=self.process.returncode,
                    files={k: str(v) for k, v in self.paths.items() if k != "stop"},
                    interpretation="CPU equivalents = scheduled CPU-seconds / wall-seconds, including spin waits. "
                    "Active threads consumed a clock tick in a sample; runnable threads may be running or waiting "
                    "for a CPU. last_cpu is a thread's last observed CPU, not simultaneous core occupancy. "
                    "Cross-phase sampling intervals are labelled boundary and excluded from phase peaks. "
                    "Phase averages use process CPU clocks and include all process threads, excluding the sampler.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--interval", type=float, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads-output", type=Path, required=True)
    p.add_argument("--stop", type=Path, required=True)
    p.add_argument("--capacity", type=int, required=True)
    args = p.parse_args()
    sample(args.pid, args.interval, args.events, args.output, args.threads_output, args.stop, args.capacity)
