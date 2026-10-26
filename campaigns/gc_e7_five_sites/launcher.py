#!/usr/bin/env python3
"""E7 side study: which five-fold seed of the star band gives five coherent structures (user, 2026-10-19).

The star, its band and a seed at five symmetric places share the five-fold symmetry, which the dynamics keep
(up to the mesh), so structures form and merge in fives. Three seeds, all with band edges displaced by at most
0.05 d (area-preserving map, same as E7):
  mode5tips     chi ~ sin(5 theta + psi), crests at the band points nearest the five convex corners (lobe tips);
  bulgetips     five outward bulges (theta-width 0.35) at the band points nearest the lobe tips;
  bulgevalleys  five outward bulges at the band points nearest the five valleys (minimum wall clearance).
At 0.05 d neither mode5tips nor bulgetips rolls up into five vortices directly (15 then 10 then 5, and 15-20):
the thin band's fastest mode is m = 15-17. Second round (18:10): large amplitudes, which make mode 5 nonlinear
before the short waves grow (strain in the stretched band between forming vortices suppresses them):
  mode5tipsA050 / mode5tipsA100  mode5tips with edges displaced by 0.5 d / 1.0 d;
  bulgetipsA100                  bulgetips with edges displaced by 1.0 d;
run to T = 100. bulgetips was stopped at t = 100 (its question answered); bulgevalleys was not run.
mode5tipsA050 (T = 100) and mode5tipsA100 (stopped at t = 50): five primary vortices first, then the stretched band
between them rolls up too (10-15). Third round (18:45): varicose seeds, the two edges moving in opposite normal
directions, so each sector's band gathers into one short thick lump (about 12 d long, up to 1.9 d thick, whose
fastest wavelength, about 15 d, exceeds its length) with thin necks between lumps:
  var5tipsA025 / var5tipsA045   varicose mode 5, lumps at the lobe tips, edges displaced by 0.25 d / 0.45 d.
var5tipsA045 gives exactly five coherent vortices (t >= 15). Fourth round (19:00): var5tipsA045p6, the same seed
at the E7 resolution (p = 6, T = 200, 24 GB free before it starts), then var5tipsA025.
Exploration resolution: the embedded CG6 star mesh, p = 4, SI-BDF2, dt = 0.05, T = 200, compatible drift;
per-step band-mode spectra, snapshots, movie. Shares the GPU with the E7 campaign (no lock; waits for
MIN_FREE_MIB free).

Start (see this campaign README; originally from the output directory):
  setsid nohup taskset -c 0-11 ./launcher.py > driver_stdout.log 2>&1 < /dev/null &
Stop: kill -- -<pgid> (pgid in driver.json). Plan only: ./launcher.py --plan. Completion marker: X5_DONE.
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

STUDY = output_directory('e7_five_sites_20261019')
HYBRIDGE = REPO
sys.path.insert(0, str(HYBRIDGE))
from campaigns.common.protocol import PYTHON, Campaign, log  # noqa: E402

RUNS = STUDY / "runs"
MARKER = STUDY / "X5_DONE"
TRIANGLES = 264268
ORDER, DT, FINAL_TIME = 4, 0.05, 200.0
POISSON_SOLVER_ATOL = 5e-14
MIN_FREE_MIB = 12000
TIPS = "1.1418,0.3691,0.0002,1.2,-1.1417,0.3695,-0.7062,-0.9702,0.7058,-0.9705"
VALLEYS = "0.4694,0.6478,-0.4691,0.6481,-0.7608,-0.2474,0.0018,-0.8,0.7607,-0.2478"
SEEDS = {
    "mode5tips": ["--modes", "star=5", "--wall-features", "star=0.0002,1.2"],
    "bulgetips": ["--sites", f"star={TIPS}", "--site-width", "0.35"],
    "bulgevalleys": ["--sites", f"star={VALLEYS}", "--site-width", "0.35"],
    "mode5tipsA050": ["--modes", "star=5", "--wall-features", "star=0.0002,1.2", "--edge-fraction", "0.5"],
    "mode5tipsA100": ["--modes", "star=5", "--wall-features", "star=0.0002,1.2", "--edge-fraction", "1.0"],
    "bulgetipsA100": ["--sites", f"star={TIPS}", "--site-width", "0.35", "--edge-fraction", "1.0"],
    "var5tipsA025": ["--modes", "star=5", "--wall-features", "star=0.0002,1.2", "--varicose", "--edge-fraction", "0.25"],
    "var5tipsA045": ["--modes", "star=5", "--wall-features", "star=0.0002,1.2", "--varicose", "--edge-fraction", "0.45"],
}
SEEDS["var5tipsA045p6"] = SEEDS["var5tipsA045"]
RUN = ("var5tipsA045p6", "var5tipsA025")  # this launch
FINAL_TIMES = {name: 100.0 for name in ("mode5tipsA050", "mode5tipsA100", "bulgetipsA100", "var5tipsA025", "var5tipsA045")}
ORDERS = {"var5tipsA045p6": 6}
MIN_FREE = {"var5tipsA045p6": 24000}


def command(name: str) -> list[str]:
    return [str(PYTHON), "-u", "-m", "scripts.guiding_center.studies.band_evolution.run_band_evolution",
            "--geometries", "star", "--kinds", "seed", *SEEDS[name], "--order", str(ORDERS.get(name, ORDER)), "--dt", f"{DT:g}",
            "--final-time", f"{FINAL_TIMES.get(name, FINAL_TIME):g}", "--poisson-solver-atol", f"{POISSON_SOLVER_ATOL:g}",
            "--output-dir", str(RUNS), "--prefix", f"x5{name}", "--skip-existing", "--verbosity", "1"]


def main() -> None:
    if "--plan" in sys.argv[1:]:
        for name in RUN:
            print(name, " ".join(command(name)))
        return
    os.chdir(STUDY)
    RUNS.mkdir(exist_ok=True)
    pgid = os.getpgid(0)
    (STUDY / "driver.json").write_text(json.dumps(dict(pid=os.getpid(), pgid=pgid, stop=f"kill -- -{pgid}",
                                                       started=datetime.now().isoformat(timespec="seconds")), indent=2) + "\n")
    campaign = Campaign(STUDY, cpu_cores="0-11", run_timeout=4 * 3600.0, max_attempts=1,
                        exclusive=False, min_free_mib=MIN_FREE_MIB)
    campaign.provenance(order=ORDER, dt=DT, final_time=FINAL_TIME, seeds=SEEDS, poisson_solver_atol=POISSON_SOLVER_ATOL,
                        min_free_mib=MIN_FREE_MIB)
    for name in RUN:
        run = dict(run_id=name, geometry="star", kind="seed", triangles=TRIANGLES, order=ORDERS.get(name, ORDER), dt=DT,
                   final_time=FINAL_TIMES.get(name, FINAL_TIME))
        campaign.min_free_mib = MIN_FREE.get(name, MIN_FREE_MIB)
        while not campaign.finished(name):
            campaign.execute(run, command(name), dict(triangles=TRIANGLES), output_glob=f"x5{name}_star_*",
                             timings_glob=f"x5{name}_star_*_timings.jsonl")
    MARKER.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
    log("five-site study done")


if __name__ == "__main__":
    main()
