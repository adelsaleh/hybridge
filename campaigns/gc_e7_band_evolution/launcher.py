#!/usr/bin/env python3
"""E7: control and band-mode seeded evolution of the four Figure 14 equilibria (PLAN.md E7, plan of 2026-10-19).

A 20-step smoke pair on the star (runs/smoke, run ids smoke_star_*) checks the device path, memory and
step time first; the production runs start only if both smoke runs succeed.
One process of scripts/guiding_center/studies/band_evolution/run_band_evolution.py per run (geometry x {control, seed}),
geometry-major so that complete pairs appear early. Accuracy campaign: the GPU lock is never taken; before each
run the protocol waits until no performance campaign holds it and the device has MIN_FREE_MIB free
(campaign_protocol.Campaign(exclusive=False)). Resume: start again; finished runs are skipped.

Start (see this campaign README; originally from the output directory):
  setsid nohup taskset -c 12-23 ./launcher.py > driver_stdout.log 2>&1 < /dev/null &
Stop: kill -- -<pgid> (pgid in driver.json). Plan only: ./launcher.py --plan. Completion marker: E7_DONE.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent

import json
import os
import sys
from datetime import datetime
from pathlib import Path

STUDY = output_directory('e7_band_evolution_20261019')
HYBRIDGE = REPO
sys.path.insert(0, str(HYBRIDGE))
from campaigns.common.protocol import PYTHON, Campaign, log  # noqa: E402

RUNS = STUDY / "runs"
MARKER = STUDY / "E7_DONE"
GEOMETRIES = ("star", "uglypacman", "iter", "horseshoe")
TRIANGLES = {"star": 264268, "uglypacman": 215246, "iter": 250997, "horseshoe": 257368}
MODES = {"star": 16, "uglypacman": 16, "iter": 18, "horseshoe": 20}
ORDER, DT, FINAL_TIME = 6, 0.05, 200.0
EDGE_FRACTION = 0.05
INITIAL_PROJECTION_QUAD_1D = 24
# max(atol, 1e-11 |b|): |b| ~ 1.1e-3 on these bands, so rtol alone (1.1e-14) sits under the round-off
# floor of the first solve (1.5e-14, star smoke 2026-10-19); 5e-14 is 4.5e-11 relative.
POISSON_SOLVER_ATOL = 5e-14
# ... and on ITER |b| ~ 9.8e-3, so rtol governs: the solve stalls at 1.46e-11 of |b| (the same relative floor as the
# star, 1.36e-11): rtol 5e-11 (iter_control 2026-10-19 18:25; star/Pac-Man ran at an effective 4.5e-11).
POISSON_SOLVER_RTOL = 5e-11
MIN_FREE_MIB = 24000
RUN_TIMEOUT = 8 * 3600.0
MAX_ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 2
CPU_CORES = "12-23"
PREFIX = "e7"


SMOKE = dict(geometry="star", steps=20, snapshot_times="0.5,1")


def smoke_runs() -> list[dict]:
    return [dict(run_id=f"smoke_star_{kind}", geometry="star", kind=kind, mode=MODES["star"] if kind == "seed" else None,
                 triangles=TRIANGLES["star"], order=ORDER, dt=DT, steps=SMOKE["steps"], smoke=True)
            for kind in ("control", "seed")]


def runs() -> list[dict]:
    return [dict(run_id=f"{geometry}_{kind}", geometry=geometry, kind=kind, mode=MODES[geometry] if kind == "seed" else None,
                 triangles=TRIANGLES[geometry], order=ORDER, dt=DT, final_time=FINAL_TIME)
            for geometry in GEOMETRIES for kind in ("control", "seed")]


def command(run: dict) -> list[str]:
    if run.get("smoke"):
        return [*command(dict(run, smoke=False))[:-7], "--steps", str(run["steps"]), "--snapshot-times",
                SMOKE["snapshot_times"], "--output-dir", str(RUNS / "smoke"), "--prefix", "e7smoke",
                "--verbosity", "2"]
    return [str(PYTHON), "-u", "-m", "scripts.guiding_center.studies.band_evolution.run_band_evolution",
            "--geometries", run["geometry"], "--kinds", run["kind"],
            "--modes", ";".join(f"{k}={v}" for k, v in MODES.items()), "--edge-fraction", f"{EDGE_FRACTION:g}",
            "--order", str(ORDER), "--dt", f"{DT:g}", "--final-time", f"{FINAL_TIME:g}",
            "--initial-projection-quad-1d", str(INITIAL_PROJECTION_QUAD_1D),
            "--poisson-solver-atol", f"{POISSON_SOLVER_ATOL:g}", "--poisson-solver-rtol", f"{POISSON_SOLVER_RTOL:g}",
            "--output-dir", str(RUNS), "--prefix", PREFIX, "--skip-existing", "--verbosity", "1"]


def main() -> None:
    if "--plan" in sys.argv[1:]:
        for run in smoke_runs() + runs():
            print(run["run_id"], " ".join(command(run)))
        return
    os.chdir(STUDY)
    RUNS.mkdir(exist_ok=True)
    pgid = os.getpgid(0)
    (STUDY / "driver.json").write_text(json.dumps(dict(pid=os.getpid(), pgid=pgid, stop=f"kill -- -{pgid}",
                                                       started=datetime.now().isoformat(timespec="seconds")), indent=2) + "\n")
    campaign = Campaign(STUDY, cpu_cores=CPU_CORES, run_timeout=RUN_TIMEOUT, max_attempts=MAX_ATTEMPTS,
                        exclusive=False, min_free_mib=MIN_FREE_MIB)
    campaign.provenance(order=ORDER, dt=DT, final_time=FINAL_TIME, modes=MODES, edge_fraction=EDGE_FRACTION,
                        initial_projection_quad_1d=INITIAL_PROJECTION_QUAD_1D, poisson_solver_atol=POISSON_SOLVER_ATOL,
                        poisson_solver_rtol=POISSON_SOLVER_RTOL,
                        min_free_mib=MIN_FREE_MIB)
    for run in smoke_runs():
        while not campaign.finished(run["run_id"]):
            row = campaign.execute(run, command(run), dict(triangles=run["triangles"]),
                                   output_glob=f"smoke/e7smoke_star_*_{run['kind']}*",
                                   timings_glob=f"smoke/e7smoke_star_*_{run['kind']}_timings.jsonl")
            if row["status"] != "ok":
                log(f"smoke run {run['run_id']} {row['status']}: stopping before production (see runs/logs)")
                (STUDY / "E7_FAILED").write_text(f"smoke {run['run_id']} {row['status']}\n")
                return
    failures = 0
    for run in runs():
        while not campaign.finished(run["run_id"]):
            row = campaign.execute(run, command(run), dict(triangles=run["triangles"]),
                                   output_glob=f"{PREFIX}_{run['geometry']}_*_{run['kind']}*",
                                   timings_glob=f"{PREFIX}_{run['geometry']}_*_{run['kind']}_timings.jsonl")
            failures = failures + 1 if row["status"] != "ok" else 0
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log(f"{failures} consecutive failed runs: stopping (see runs/logs); restart resumes")
                (STUDY / "E7_FAILED").write_text(datetime.now().isoformat(timespec="seconds") + "\n")
                return
    MARKER.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
    log("E7 done")


if __name__ == "__main__":
    main()
