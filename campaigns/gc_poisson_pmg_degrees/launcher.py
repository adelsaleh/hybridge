#!/usr/bin/env python3
"""Poisson p-multigrid degree study: fb-hp-mg-pcg against the AMGX candidates at p = 5..9, L2 and L3.

Performance study (AGENTS.md): it holds hybridge/.cache/gpu-campaign.lock for its whole duration and
starts every run on an idle GPU (protocol: campaigns/common/protocol.py). Each
run is one process of scripts/diffusion_reaction/benchmark_poisson_pmg_degrees.py for one (p, level):
the campaign's Poisson system (disk-highmode m = 128, tau = 128000, mesh_levels_smoke.jsonl of the D8-S
smoke campaign), candidates fbmg-standard, amgx-cheb-csr, amgx-classical-abs-csr; cold solve, zero-guess
repeat, 10 warm solves of perturbed right-hand sides (linear phase), on L2 also 10 warm solves of the
saturated-phase sequence rho_eq + s_k (rho_sat - rho_eq) from the stage H snapshot (warm_saturated), peak
device memory, solution agreement. Order:
p = 5..9 at L2, the tau = 1 sanity checks of the native solver (p = 6 and 9 at L2), then p = 5..9 at L3.

Start detached (see this campaign README; originally from the output directory):
  setsid nohup taskset -c 12-23 ./launcher.py > driver_stdout.log 2>&1 < /dev/null &
Stop: kill -- -<pgid>. Resume: start it again; finished runs (runs.jsonl) are skipped, and inside a
run the benchmark skips candidates already in runs/results.jsonl. Completion marker: POISSON_PMG_DONE.
Results: runs/results.jsonl (one row per (p, level, candidate)), runs/summary.md, runs/<run>.log,
runs/logs/<run_id>.log (driver copy), runs/gpu/*.csv (nvidia-smi samples), runs.jsonl (protocol records).
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

STUDY = output_directory('poisson_pmg_degrees_20261017')
HYBRIDGE = REPO
sys.path.insert(0, str(HYBRIDGE))
from campaigns.common.protocol import PYTHON, Campaign, log  # noqa: E402

RUNS = STUDY / "runs"
MARKER = STUDY / "POISSON_PMG_DONE"
MESH_LEVELS = record_file(ARTICLE_OUTPUTS / "diocotron_order_smoke_20261016", CODE.parent / "gc_d8s_smoke", "mesh_levels_smoke.jsonl")
FAMILY = "m128"
TAU = 128000.0
CANDIDATES = "fbmg-standard,amgx-cheb-csr,amgx-classical-abs-csr"
WARM_SOLVES = 10
DEGREES = (5, 6, 7, 8, 9)
RUN_TIMEOUT = 1800.0  # seconds per (p, level) process
MAX_ATTEMPTS = 2  # a contaminated run is queued once more
CPU_CORES = "12-23"
# tau = 1 sanity checks of the native solver (the audit attributes the weak p = 0 coarse level to tau).
TAU_ONE_CHECKS = ((6, "L2"), (9, "L2"))


def mesh_levels() -> dict:
    levels = {}
    for line in MESH_LEVELS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["family"] == FAMILY:
                levels[(row["level"], int(row["p"]))] = row
    return levels


def specs() -> list[dict]:
    out = [dict(run_id=f"L2_p{p}", level="L2", p=p, tau=TAU, candidates=CANDIDATES, tag="") for p in DEGREES]
    out += [dict(run_id=f"{level}_p{p}_tau1", level=level, p=p, tau=1.0, candidates="fbmg-standard", tag="tau1")
            for p, level in TAU_ONE_CHECKS]
    out += [dict(run_id=f"L3_p{p}", level="L3", p=p, tau=TAU, candidates=CANDIDATES, tag="") for p in DEGREES]
    return out


def command(run: dict) -> list[str]:
    args = [str(PYTHON), "-u", "-m", "scripts.diffusion_reaction.benchmark_poisson_pmg_degrees",
            "--degrees", str(run["p"]), "--levels", run["level"], "--candidates", run["candidates"],
            "--warm-solves", str(WARM_SOLVES), "--tau", f"{run['tau']:g}", "--mesh-levels", str(MESH_LEVELS),
            "--family", FAMILY, "--output-dir", str(RUNS), "--verbosity", "3"]
    if run["tag"]:
        args += ["--run-tag", run["tag"]]
    return args


def main() -> None:
    if sys.argv[1:] == ["--plan"]:
        levels = mesh_levels()
        for run in specs():
            level = levels.get((run["level"], run["p"]))
            print(run["run_id"], "" if level is None else f"{level['triangles']} triangles")
            print("  ", " ".join(command(run)))
        return
    os.chdir(STUDY)
    RUNS.mkdir(exist_ok=True)
    campaign = Campaign(STUDY, cpu_cores=CPU_CORES, run_timeout=RUN_TIMEOUT, max_attempts=MAX_ATTEMPTS)
    campaign.hold_lock()
    try:
        campaign.provenance(tau=TAU, candidates=CANDIDATES, warm_solves=WARM_SOLVES, degrees=list(DEGREES))
        levels = mesh_levels()
        for run in specs():
            while not campaign.finished(run["run_id"]):
                level = levels.get((run["level"], run["p"]))
                if level is None:
                    log(f"no mesh level for {run['run_id']}; skipped")
                    campaign.append(dict(run, status="no-mesh", attempt=1))
                    continue
                campaign.execute(run, command(run), level, output_glob=f"{run['level']}_p{run['p']}*.log")
        MARKER.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
        log("Poisson pMG degree study done")
    finally:
        campaign.release()


if __name__ == "__main__":
    main()
