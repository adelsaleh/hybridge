#!/usr/bin/env python3
"""E7 lump study: geometry-tied varicose perturbations on every domain (user, 2026-10-20: "pursue the search
for meaningful perturbations on each of our geometries").

On the star, a varicose mode-5 seed with crests at the five lobe tips gathers each sector's band into one
short thick lump that rolls up into one vortex: exactly five vortices, one per lobe, to T = 200
(e7_five_sites_20261019). A lump rolls up into a single vortex when it is shorter than the fastest wavelength
of its own thickness, which holds for sectors of about 25 d and edges moved apart by 0.45 d (necks 0.1 d);
a sector of about 25 d means mode 5 on the star and the Ugly Pac-Man (L/d = 127), 5-7 on ITER (139) and 5-6
on the horseshoe (154). The mode and phase are chosen so that the crests sit on the band's sharpest turns
(curvature maxima of the middle contour):
  star5valleys    mode 5, crests at the five valleys instead of the lobe tips: do the vortices move to the lobes?
  pacman5         mode 5, crest at the upper jaw turn (s = 0.00); the other crests fall near the mouth bend
                  (0.21, concave, by the re-entrant corner) and the lower jaw turn (0.38);
  pacman5mouth    mode 5, crest at the mouth bend (lumps shifted by 0.01-0.03 in s);
  iter7           mode 7, crest at the upper corner of the D (s = 0.875); the corners are 4/7 and 3/7 of the band
                  apart, so another crest falls on the lower corner by the X-point (s = 0.306);
  iter5           mode 5, crest at the lower corner by the X-point (28 d sectors);
  horseshoe5      mode 5, crest at a hairpin turn: the turns are exactly 2/5 of the band apart, so crests fall on
                  both turns, the top of the inner arc and two symmetric points of the outer arc (31 d sectors);
  horseshoe5A055  the same with edges moved apart by 0.55 d (thicker lumps for the longer sectors).
Exploration resolution as in e7_five_sites: embedded CG6 meshes, p = 4, SI-BDF2, dt = 0.05, T = 100, compatible
drift, per-step band-mode spectra, snapshots, movie. Accuracy runs: no GPU lock, two lanes share the GPU
(about 12 GB each at p = 4), each waits for MIN_FREE_MIB free device memory.
Lane c (article resolution, p = 6, T = 200, about 23 GB, waits for 30 GB free so that it starts once one of the
p = 4 lanes is done): pacman5p6, iter7p6, horseshoe5p6, the seeds of pacman5, iter7 and horseshoe5. A file
SKIP_<name> in this directory makes the lane skip that run (e.g. if its p = 4 exploration disappoints).

Start (see this campaign README; originally from the output directory), one process group per lane:
  setsid nohup taskset -c 0-11 ./launcher.py --lane a > driver_stdout_a.log 2>&1 < /dev/null &
  setsid nohup taskset -c 12-23 ./launcher.py --lane b > driver_stdout_b.log 2>&1 < /dev/null &
  setsid nohup taskset -c 12-23 ./launcher.py --lane c > driver_stdout_c.log 2>&1 < /dev/null &
Stop: kill -- -<pgid> (pgid in driver_<lane>.json). Plan only: ./launcher.py --plan. Completion marker per lane:
LUMPS_DONE_<lane>. Resuming skips finished runs (runs.jsonl).
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

STUDY = output_directory('e7_lumps_20261020')
HYBRIDGE = REPO
sys.path.insert(0, str(HYBRIDGE))
from campaigns.common.protocol import PYTHON, Campaign, log  # noqa: E402

RUNS = STUDY / "runs"
TRIANGLES = dict(star=264268, uglypacman=215246, iter=250997, horseshoe=257368)
ORDER, DT, FINAL_TIME = 4, 0.05, 100.0
POISSON_SOLVER_ATOL, POISSON_SOLVER_RTOL = 5e-14, 5e-11
MIN_FREE_MIB = 14000
# name: (geometry, mode, crest point (band point nearest to it), edge fraction)
SEEDS = {
    "star5valleys": ("star", 5, (0.0, -0.67), 0.45),
    "pacman5": ("uglypacman", 5, (-0.426, 0.629), 0.45),
    "pacman5mouth": ("uglypacman", 5, (0.426, -0.102), 0.45),
    "iter7": ("iter", 7, (1.198, 8.804), 0.45),
    "iter5": ("iter", 5, (1.361, 1.905), 0.45),
    "horseshoe5": ("horseshoe", 5, (-0.642, -0.487), 0.45),
    "horseshoe5A055": ("horseshoe", 5, (-0.642, -0.487), 0.55),
    "pacman5p6": ("uglypacman", 5, (-0.426, 0.629), 0.45),
    "iter7p6": ("iter", 7, (1.198, 8.804), 0.45),
    "horseshoe5p6": ("horseshoe", 5, (-0.642, -0.487), 0.45),
}
LANES = dict(a=("pacman5", "horseshoe5", "iter5", "star5valleys"),
             b=("iter7", "pacman5mouth", "horseshoe5A055"),
             c=("pacman5p6", "iter7p6", "horseshoe5p6"))
ORDERS: dict[str, int] = {name: 6 for name in LANES["c"]}
FINAL_TIMES: dict[str, float] = {name: 200.0 for name in LANES["c"]}
MIN_FREE: dict[str, int] = {name: 30000 for name in LANES["c"]}
# horseshoe5A055 ran out of device memory at t = 21 (13:09) as the third run on the GPU next to a p = 6 run
# (33 GB) and a p = 4 run; its rerun waits until at most one other run is on the device.
MIN_FREE["horseshoe5A055"] = 20000  # a p = 6 run alone leaves about 23 GB free; 11.4 GB needed


def command(name: str) -> list[str]:
    g, mode, crest, fraction = SEEDS[name]
    return [str(PYTHON), "-u", "-m", "scripts.guiding_center.studies.band_evolution.run_band_evolution",
            "--geometries", g, "--kinds", "seed", "--modes", f"{g}={mode}",
            "--wall-features", f"{g}={crest[0]:g},{crest[1]:g}", "--varicose", "--edge-fraction", f"{fraction:g}",
            "--order", str(ORDERS.get(name, ORDER)), "--dt", f"{DT:g}",
            "--final-time", f"{FINAL_TIMES.get(name, FINAL_TIME):g}",
            "--poisson-solver-atol", f"{POISSON_SOLVER_ATOL:g}", "--poisson-solver-rtol", f"{POISSON_SOLVER_RTOL:g}",
            "--output-dir", str(RUNS), "--prefix", f"lp{name}", "--skip-existing", "--verbosity", "1"]


def main() -> None:
    args = sys.argv[1:]
    lanes = [args[args.index("--lane") + 1]] if "--lane" in args else sorted(LANES)
    if "--plan" in args:
        for lane in lanes:
            for name in LANES[lane]:
                print(lane, name, " ".join(command(name)))
        return
    if len(lanes) != 1:
        raise SystemExit("start one lane per process: --lane a|b")
    lane = lanes[0]
    os.chdir(STUDY)
    RUNS.mkdir(exist_ok=True)
    pgid = os.getpgid(0)
    (STUDY / f"driver_{lane}.json").write_text(json.dumps(dict(pid=os.getpid(), pgid=pgid, stop=f"kill -- -{pgid}",
                                                               started=datetime.now().isoformat(timespec="seconds")),
                                                          indent=2) + "\n")
    cores = "0-11" if lane == "a" else "12-23"
    campaign = Campaign(STUDY, cpu_cores=cores, run_timeout=4 * 3600.0, max_attempts=1,
                        exclusive=False, min_free_mib=MIN_FREE_MIB)
    campaign.provenance(lane=lane, order=ORDER, dt=DT, final_time=FINAL_TIME, seeds=SEEDS, orders=ORDERS,
                        final_times=FINAL_TIMES, poisson_solver_atol=POISSON_SOLVER_ATOL,
                        poisson_solver_rtol=POISSON_SOLVER_RTOL, min_free_mib=MIN_FREE_MIB, min_free=MIN_FREE)
    for name in LANES[lane]:
        g = SEEDS[name][0]
        run = dict(run_id=name, geometry=g, kind="seed", triangles=TRIANGLES[g], order=ORDERS.get(name, ORDER),
                   dt=DT, final_time=FINAL_TIMES.get(name, FINAL_TIME))
        campaign.min_free_mib = MIN_FREE.get(name, MIN_FREE_MIB)
        if (STUDY / f"SKIP_{name}").exists():
            log(f"skip {name} (SKIP_{name} present)")
            continue
        # Campaign.finished also counts failed attempts (oom, timeout) as finished; rerun those on resume.
        if not any(row["run_id"] == name and row["status"] == "ok" for row in campaign.records()):
            campaign.execute(run, command(name), dict(triangles=TRIANGLES[g]), output_glob=f"lp{name}_{g}_*",
                             timings_glob=f"lp{name}_{g}_*_timings.jsonl")
    (STUDY / f"LUMPS_DONE_{lane}").write_text(datetime.now().isoformat(timespec="seconds") + "\n")
    log(f"lump study lane {lane} done")


if __name__ == "__main__":
    main()
