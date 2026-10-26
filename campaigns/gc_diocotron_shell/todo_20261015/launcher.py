"""Remaining diocotron-section runs (D0 l=3 diagnostics, D1 ladders, D2 dispersion, D4 long run).

One simulation per process. These are accuracy runs, not performance runs, so they
do not take the shared GPU lock (``.cache/gpu-campaign.lock``, reserved for timing
runs); each run starts when the GPU has ``MIN_FREE_MIB`` of free memory.
``--skip-existing`` makes the driver return at once for completed runs, so rerunning
this launcher resumes. Writes ``launcher.done`` when finished.

    setsid nohup .venv/bin/python <this file> > <dir>/driver_stdout.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent

import datetime
import itertools
import subprocess
import sys
from pathlib import Path

HYBRIDGE = REPO
HERE = output_directory('diocotron') / 'todo_20261015'
OUT = str(output_directory('diocotron'))  # symlink to the article's run_outputs/diocotron
MIN_FREE_MIB = 8000
DRIVER = [str(HYBRIDGE / ".venv/bin/python"), "-m", "scripts.guiding_center.studies.diocotron.run_diocotron_benchmark"]


def runs():
    """(name, prefix, driver arguments) of every simulation, one each."""
    jobs = []

    def add(prefix, base, *, orders=(6,), bands, schemes, dts, velocities=("raw",), modes=(None,),
            controls=True, extra_per_mode=None):
        for order, band, scheme, dt, velocity, mode in itertools.product(orders, bands, schemes, dts, velocities,
                                                                          modes):
            args = [*base, "--orders", str(order), "--band-mesh-sizes", str(band), "--schemes", scheme,
                    "--dts", str(dt), "--velocities", velocity]
            if mode is not None:
                args += ["--modes", str(mode)]
                if extra_per_mode:
                    args += extra_per_mode[mode]
            kinds = ("control", "seed") if controls else ("seed",)
            for kind in kinds:
                run_args = args + (["--controls"] if controls else []) + ["--only", kind]
                name = f"{prefix}_p{order}_hb{band}_{scheme}_dt{dt}_{velocity}" + (
                    f"_m{mode}" if mode is not None else "") + f"_{kind}"
                jobs.append((name, prefix, run_args))

    quiet = ["--movie-interval", "0"]
    # D0: l = 3 discrepancy on the 14,628-triangle band mesh, and an l = 4 mesh ladder.
    annulus = ["--benchmark", "annulus-sl", "--modes", "3", *quiet]
    add("d0eps", [*annulus, "--epsilon", "1e-4"], bands=(0.01,), schemes=("imex-ark3",), dts=(0.05,))
    for tau in ("320", "3200", "320000"):
        add(f"d0tau{tau}", [*annulus, "--poisson-tau", tau], bands=(0.01,), schemes=("imex-ark3",), dts=(0.05,))
    add("d0p", annulus, orders=(4, 5), bands=(0.01,), schemes=("imex-ark3",), dts=(0.05,))
    add("d0l4", ["--benchmark", "annulus-sl", "--modes", "4", *quiet], bands=(0.02, 0.01, 0.005),
        schemes=("si-bdf2", "imex-ark3"), dts=(0.05,))
    # D2: the smooth ring seeded at other modes (m = 9 is D1), with mode-dependent length and fit window.
    smooth = ["--benchmark", "disk-smooth", "--poisson-tau", "128000"]
    windows = {3: ("120", "35,90"), 5: ("80", "18,48"), 7: ("64", "15,37"), 10: ("64", "14,33"),
               11: ("64", "14,34"), 13: ("64", "16,40")}
    add("d2", smooth, bands=(0.008,), schemes=("imex-ark3",), dts=(0.05,), modes=tuple(windows),
        extra_per_mode={m: ["--final-time", t, "--fit-window", w] for m, (t, w) in windows.items()})
    # D1: p ladder, time-step ladder, Poisson tau (raw drift), sign of the seed.
    add("d1p", [*smooth, *quiet], orders=(4, 5), bands=(0.016, 0.008, 0.004), schemes=("si-bdf2", "imex-ark3"),
        dts=(0.05,))
    add("d1dt", [*smooth, *quiet], bands=(0.008,), schemes=("si-bdf2", "imex-ark3"), dts=(0.8, 0.4, 0.2, 0.1))
    for tau in ("128", "1280", "12800"):
        add(f"d1tau{tau}", ["--benchmark", "disk-smooth", "--poisson-tau", tau, *quiet], bands=(0.008,),
            schemes=("imex-ark3",), dts=(0.05,))
    # "--epsilon=": argparse reads a bare -1e-4 as an option.
    add("d1neg", [*smooth, "--epsilon=-1e-4", *quiet], bands=(0.008,), schemes=("si-bdf2", "imex-ark3"),
        dts=(0.05,))
    # D4: smooth m = 9 ring through the nonlinear phase, snapshots and movie.
    add("d4", [*smooth, "--final-time", "150", "--snapshot-times", "20,30,35,40,45,50,55,60,70,80,100,120,150",
               "--enstrophy-budget"],
        bands=(0.008,), schemes=("imex-ark3", "si-bdf2"), dts=(0.05,), controls=False)
    return jobs


def log(message: str) -> None:
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (HERE / "launcher.log").open("a") as stream:
        stream.write(line + "\n")


def free_gpu_mib() -> int:
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    return min(int(line) for line in output.split() if line.strip())


def wait_for_memory() -> None:
    from campaigns.common.protocol import wait_for_free_device

    wait_for_free_device(MIN_FREE_MIB)


def main() -> None:
    jobs = runs()
    prefixes = [arg.split("=", 1)[1].split(",") for arg in sys.argv if arg.startswith("--prefixes=")]
    if prefixes:  # rerun a subset, e.g. --prefixes=d1neg
        jobs = [job for job in jobs if job[1] in prefixes[0]]
    if "--list" in sys.argv:
        for name, prefix, args in jobs:
            print(name, " ".join(args))
        print(len(jobs), "runs")
        return
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    log(f"start: {len(jobs)} runs")
    failures = 0
    for index, (name, prefix, args) in enumerate(jobs, 1):
        command = [*DRIVER, *args, "--output-dir", OUT, "--prefix", prefix, "--skip-existing"]
        wait_for_memory()
        log(f"[{index}/{len(jobs)}] {name}: start")
        with (HERE / "logs" / f"{name}.log").open("a") as stream:
            status = subprocess.run(command, cwd=HYBRIDGE, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}).returncode
        failures += status != 0
        log(f"[{index}/{len(jobs)}] {name}: exit {status}")
    (HERE / "launcher.done").write_text(f"{datetime.datetime.now().isoformat(timespec='seconds')} "
                                        f"{len(jobs)} runs, {failures} nonzero exits\n")
    log("done")


if __name__ == "__main__":
    main()
