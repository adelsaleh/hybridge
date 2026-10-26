#!/usr/bin/env python3
"""D8-S smoke campaign: fastest correct device preset per degree p = 5..9 for the D8 diocotron campaign.

Performance campaign (AGENTS.md): it holds hybridge/.cache/gpu-campaign.lock for its whole duration
and starts every run on an idle GPU (protocol: campaigns/common/protocol.py).
Each run is one process of the diocotron driver
(scripts/guiding_center/studies/diocotron/run_diocotron_benchmark.py): a few SI-BDF2 steps on a very large mesh
(9 M element unknowns, h scaled with p), synchronized step timing, every record kept. Design: PLAN.md,
D8, "Smoke campaign D8-S". Restricted to p = 5..9 on 2026-10-16 (user: D8 keeps no degree below 5); the
finished p = 1, 2, 4 runs stay in runs.jsonl.

Stages (each selects before the next; selections in presets_selected.json):
  A  Poisson solver candidates per p (raw drift, default transport)
  B  Poisson kernel variants around the winner (cache policy, trace basis, block size)
  C  transport variants with the Poisson winner
  H  the saturated phase: the two fastest Poisson bases per p on L2, seed 0.1, 1000 steps; the Poisson
     preset is then chosen by the step time over t >= 15, where the extrapolated potential is a poor guess
  D  drift: raw with Poisson degree p+1, and V2 (compatible drift, p+1 postprocessed potential)
  E  volume quadrature: the collapsed rule exact for the advection term ((3p+2)//2 points per direction)
     against the default, at every p
  F  memory and step time at the m = 128 reference level (27 M element unknowns)
  G  the selected presets on the Zoni-Guclu disk (9 M), with one deliberate foreign GPU process (must be flagged)

Start detached (see this campaign README; originally from the output directory):
  setsid nohup taskset -c 12-23 ./launcher.py > driver_stdout.log 2>&1 < /dev/null &
Stop: kill -- -<pgid>. Resume: start it again; finished runs (runs.jsonl) are skipped.
Completion marker: SMOKE_DONE. Host process groups listed in pause_pgids.json (read before every run) are
paused with SIGSTOP during each run and resumed with SIGCONT after it and on exit.
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
import random
import sys
from datetime import datetime
from pathlib import Path

STUDY = output_directory('diocotron_order_smoke_20261016')
HYBRIDGE = REPO
sys.path.insert(0, str(HYBRIDGE))
from campaigns.common.protocol import PYTHON, Campaign, log  # noqa: E402
from scripts.guiding_center.studies.diocotron.analyze_diocotron_order import (  # noqa: E402
    select_presets, smoke_report)
from scripts.guiding_center.runtime.fast_profiles import default_transport_candidate  # noqa: E402

RUNS = STUDY / "runs"
MARKER = STUDY / "SMOKE_DONE"
# Wait for these before taking the lock (the convergence session's exclusive timing hour).
WAIT_FOR = [(ARTICLE_OUTPUTS / 'convergence_poisson_degree/timing_device/DONE')]
DT, STEPS, REFERENCE_STEPS = 0.025, 8, 4
TAU = {"m128": 128000.0, "zg": 32000.0}
BENCHMARK = {"m128": ["--benchmark", "disk-highmode", "--modes", "128"], "zg": ["--benchmark", "disk-zg"]}
SEED = 20261016
MAX_ATTEMPTS = 2  # a contaminated run is queued once more
MAX_CONSECUTIVE_FAILURES = 4  # then stop: probably a defect of the tooling, not of a candidate
RUN_TIMEOUT = 1800.0  # seconds; a run still going then is stopped and recorded as "timeout"
# Runs are confined to these cores: an MPI study of another session binds its ranks to cores 0-11 (2026-10-16).
CPU_CORES = "12-23"

DEGREES = range(5, 10)
# amgx-classical-abs-csr (added 2026-10-16 after the pMG audit): the amgx-classical-csr config stops at 1e-13
# of the initial residual and over-solves every warm step; the -abs variant stops at an absolute 1e-12.
STAGE_A = {
    **{p: ["fbmg-standard", "fbmg-fast", "amgx-cheb-csr", "amgx-cheb-bsr", "amgx-classical-abs-csr"] for p in (5, 6)},
    # fbmg at p = 7-9 added 2026-10-17 after the pure Poisson test (poisson_pmg_degrees_20261017): the
    # fb-hp-mg degree gate was validation scope only, and the native solver had the cheapest warm solves at
    # every p = 5-9 there (user: "the native pMG-AMG should also be tested in the campaign").
    **{p: ["amgx-cheb-csr", "amgx-classical-csr", "amgx-cheb-bsr", "amgx-classical-abs-csr", "fbmg-standard",
           "fbmg-fast"] for p in (7, 8, 9)},
}
STAGE_B = {6: ["lu"], 8: ["modal"], 9: ["lu", "bs64", "bs128"]}
STAGE_C = {5: ["fused+bsr+bj"], 6: ["split3+bsr", "fused+bsr+bj"], 7: ["split3+bsr", "fused+bsr+bj", "split3+bsr+bj"],
           8: ["fused+bsr", "split3+bsr+bj", "split3+bsr+l1"], 9: ["fused+bsr", "split3+bsr+bj"]}
# Stage E: the collapsed volume rule with n^2 points (exact to degree 2n-2) that is exact for the advection
# term with a degree-p drift (degree 3p-1: n = (3p+2)//2) against the default rule (Dunavant 2p at p <= 7,
# which under-integrates that term; (2p+2)^2 collapsed points at p >= 8, exact, operator by reference
# tensors). One exactness policy for every degree of D8; at p <= 7 it moves the kernel to the tensor path
# (kernel audit 2026-10-16: about +16 % assembly flops at p = 7), at p >= 8 it halves the diagnostics' points.
STAGE_E = {p: [(3 * p + 2) // 2] for p in DEGREES}
# Stage H: the saturated phase. Seed 0.1 saturates at t of about 15 (growth rate 0.185) instead of 50;
# 1000 steps of 0.025 on L2 (3 M element unknowns); diagnostics every 10 steps; medians over t >= 15 in
# analyze_diocotron_order.
NONLINEAR = dict(level="L2", epsilon=0.1, steps=1000, diagnostics_every=10, candidates=2)
# Stage J (added 2026-10-17 after the campaign): the Poisson stopping rule. The runs above stop the Poisson
# solve at the absolute residual 1e-10 while the reduced right-hand side has norm 4-8e-5 (relative 1e-6;
# runs of the same discrete problem with different Poisson solvers differ by 3-7e-7 in the final density).
# D8 needs a floor well below its discretization errors: the selected presets with --poisson-solver-atol
# 1e-14 (relative about 2e-10) on L3 at p = 5, 7, 9 (step cost and warm iterations), the runner-up Poisson
# candidate at p = 7 (agreement at the tight rule), and the saturated phase at p = 7 and 9 (iteration growth).
STAGE_J = dict(atol=1e-14, degrees=(5, 7, 9), agreement_degree=7, nonlinear_degrees=(7, 9))
# "B" removed from the resume of 2026-10-17: its five runs are done (modal trace, dense LU and block-size
# overrides gave no gain) and its run ids would otherwise be rebuilt around the later-chosen transport.
# "H" removed 2026-10-17 15:20 (its 16 runs are done; the selection reads them from the records): with the
# Poisson winner changed to fbmg-standard at p = 7-9 by stage H, stage C must add the block-Jacobi runs for
# it before D, E, G and J run with the final presets, and H's run ids would otherwise be rebuilt around
# the new transport.
STAGES = ("A", "A2", "C", "D", "E", "F", "G", "J")


def slug(text: str) -> str:
    return text.replace("+", "-").replace(".", "p")


def mesh_levels() -> dict:
    rows = [json.loads(line) for line in (record_file(STUDY, CODE, "mesh_levels_smoke.jsonl")).read_text().splitlines() if line.strip()]
    return {(row["family"], row["level"], row["p"]): row for row in rows}


def spec(stage, family, level, p, poisson, transport=None, *, velocity="raw", offset=0, vq=None, steps=STEPS,
         foreign=False, epsilon=None, diagnostics_every=1, atol=None):
    transport = transport or default_transport_candidate(p)
    variant = {"raw": f"raw{'q' + str(p + offset) if offset else ''}", "compatible": "v2"}[velocity]
    variant += f"_vq{vq}" if vq else ""
    variant += f"_eps{slug(f'{epsilon:g}')}" if epsilon is not None else ""
    variant += f"_atol{slug(f'{atol:g}')}" if atol is not None else ""
    run_id = f"{stage}_{family}{level}_p{p}_{slug(poisson)}_{slug(transport)}_{variant}"
    return dict(run_id=run_id, stage=stage, family=family, level=level, p=p, q=p + offset, poisson=poisson,
                transport=transport, velocity=velocity, offset=offset, vq=vq, steps=steps, dt=DT, foreign=foreign,
                epsilon=epsilon, diagnostics_every=diagnostics_every, atol=atol)


def top_poisson(choice: dict, count: int) -> list[str]:
    """The best variant of each of the ``count`` fastest Poisson bases (by step time) of a selection's
    stage A/B ranking, e.g. ["amgx-cheb-csr+lu", "fbmg-fast"]."""
    best = {}
    for name, step, _hard in choice.get("poisson_ranking", []):
        base = name.split("+")[0]
        if base not in best or step < best[base][1]:
            best[base] = (name, step)
    return [name for name, _ in sorted(best.values(), key=lambda item: item[1])[:count]]


def stage_specs(stage: str, selected: dict, done: list[dict] = ()) -> list[dict]:
    """Run specifications of one stage, given the selections made so far (keys are str(p)) and the
    launcher records so far (for the out-of-memory fallback)."""
    def winner(p):
        return selected.get(str(p), {})

    def level(p):
        return winner(p).get("level", "L3")

    if stage == "A":
        return [spec("A", "m128", "L3", p, name) for p, names in STAGE_A.items() for name in names]
    if stage == "A2":  # a degree whose L3 runs all ran out of device memory is selected on L2
        out_of_memory = {row["p"] for row in done if row["stage"] == "A" and row["status"] == "oom"}
        return [spec("A2", "m128", "L2", p, name) for p in sorted(out_of_memory) if str(p) not in selected
                for name in STAGE_A[p]]
    if stage == "B":
        out = []
        for p, modifiers in STAGE_B.items():
            base = winner(p).get("poisson")
            if base is None:
                continue
            for modifier in modifiers:
                if modifier == "modal" and (base.startswith("fbmg") or "modal" in base):
                    continue
                if modifier.startswith("bs") and base.startswith("fbmg"):  # AMGX block-size overrides only
                    continue
                out.append(spec("B", "m128", level(p), p, f"{base}+{modifier}", winner(p).get("transport")))
        return out
    if stage == "C":
        return [spec("C", "m128", level(p), p, winner(p)["poisson"], transport)
                for p, transports in STAGE_C.items() if winner(p).get("poisson") for transport in transports]
    if stage == "H":
        return [spec("H", "m128", NONLINEAR["level"], p, poisson, winner(p)["transport"], epsilon=NONLINEAR["epsilon"],
                     steps=NONLINEAR["steps"], diagnostics_every=NONLINEAR["diagnostics_every"])
                for p in DEGREES if winner(p).get("poisson") for poisson in top_poisson(winner(p), NONLINEAR["candidates"])]
    if stage == "D":
        out = []
        for p in DEGREES:
            if not winner(p).get("poisson"):
                continue
            out.append(spec("D", "m128", level(p), p, winner(p)["poisson"], winner(p)["transport"],
                            velocity="compatible"))
            if p + 1 <= 9 and winner(p + 1).get("poisson"):
                out.append(spec("D", "m128", level(p), p, winner(p + 1)["poisson"], winner(p)["transport"], offset=1))
        return out
    if stage == "E":
        return [spec("E", "m128", level(p), p, winner(p)["poisson"], winner(p)["transport"], vq=vq)
                for p, counts in STAGE_E.items() if winner(p).get("poisson") for vq in counts]
    if stage == "F":
        # p = 8 dropped 2026-10-17: the L4 mesh (598,613 triangles) ran out of device memory with every
        # preset (45.9 GB raw, 44.6 GB V2 after the first step); L3 p = 8 already takes 27 GB raw, 36 GB V2.
        return [spec("F", "m128", "L4", p, winner(p)["poisson"], winner(p)["transport"], steps=REFERENCE_STEPS)
                for p in (5,) if winner(p).get("poisson")]
    if stage == "G":
        return [spec("G", "zg", "L4", p, winner(p)["poisson"], winner(p)["transport"], foreign=(p == 9))
                for p in DEGREES if winner(p).get("poisson")]
    if stage == "J":
        atol = STAGE_J["atol"]
        out = [spec("J", "m128", level(p), p, winner(p)["poisson"], winner(p)["transport"], atol=atol)
               for p in STAGE_J["degrees"] if winner(p).get("poisson")]
        p = STAGE_J["agreement_degree"]
        runner_up = [name for name in top_poisson(winner(p), 2) if name != winner(p).get("poisson")]
        if winner(p).get("poisson") and runner_up:
            out.append(spec("J", "m128", level(p), p, runner_up[0], winner(p)["transport"], atol=atol))
        out += [spec("J", "m128", NONLINEAR["level"], p, winner(p)["poisson"], winner(p)["transport"],
                     epsilon=NONLINEAR["epsilon"], steps=NONLINEAR["steps"],
                     diagnostics_every=NONLINEAR["diagnostics_every"], atol=atol)
                for p in STAGE_J["nonlinear_degrees"] if winner(p).get("poisson")]
        return out
    raise ValueError(stage)


def command(run: dict, level: dict) -> list[str]:
    final_time = run["steps"] * run["dt"]
    args = [str(PYTHON), "-u", "-m", "scripts.guiding_center.studies.diocotron.run_diocotron_benchmark",
            *BENCHMARK[run["family"]], "--orders", str(run["p"]), "--band-mesh-sizes", level["hb"],
            "--mesh-size", level["h"], "--dts", f"{run['dt']:g}", "--final-time", f"{final_time:g}",
            "--schemes", "si-bdf2", "--velocities", run["velocity"], "--poisson-tau", f"{TAU[run['family']]:g}",
            "--poisson-candidate", run["poisson"], "--transport-candidate", run["transport"],
            "--poisson-order-offset", str(run["offset"]), "--synchronize-step-timing", "--enstrophy-budget",
            "--movie-interval", "0", "--snapshot-times", f"{final_time:g}", "--snapshot-density-only",
            "--diagnostics-every", str(run.get("diagnostics_every", 1)),
            "--output-dir", str(RUNS), "--prefix", "d8s", "--run-tag", run["run_id"], "--verbosity", "3"]
    if run["velocity"] == "compatible":
        args += ["--compatible-potential", "postprocessed"]
    if run["vq"]:
        args += ["--volume-quad-1d", str(run["vq"])]
    if run.get("epsilon") is not None:
        args += ["--epsilon", f"{run['epsilon']:g}"]
    if run.get("atol") is not None:
        args += ["--poisson-solver-atol", f"{run['atol']:g}"]
    return args


def resolve_level(run: dict, levels: dict) -> tuple[dict, dict | None]:
    """The mesh row of a run; the Zoni-Guclu L4 falls back to L3 where gmsh failed (p = 1, 2)."""
    level = levels.get((run["family"], run["level"], run["p"]))
    if level is None and run["family"] == "zg" and run["level"] == "L4":
        level = levels.get(("zg", "L3", run["p"]))
        if level is not None:
            run = dict(run, level="L3")
            log(f"{run['run_id']}: no ZG L4 mesh, using L3")
    return run, level


def plan() -> None:
    """Print every stage's runs and commands for a plausible selection (no lock, no GPU)."""
    levels = mesh_levels()
    selected = {str(p): dict(poisson=STAGE_A[p][0], transport=default_transport_candidate(p), level="L3",
                             poisson_ranking=[(name, 0.0, 0.0) for name in STAGE_A[p]])
                for p in DEGREES}
    total = 0
    for stage in STAGES:
        runs = stage_specs(stage, selected)
        total += len(runs)
        print(f"stage {stage}: {len(runs)} runs")
        for run in runs:
            level = levels.get((run["family"], run["level"], run["p"]))
            print("  ", run["run_id"], "" if level is None else f"{level['triangles']} triangles")
    print(f"{total} runs")
    run = stage_specs("D", selected)[0]
    print(" ".join(command(run, levels[(run["family"], run["level"], run["p"])])))


def main() -> None:
    if sys.argv[1:] == ["--plan"]:
        plan()
        return
    os.chdir(STUDY)
    RUNS.mkdir(exist_ok=True)
    campaign = Campaign(STUDY, cpu_cores=CPU_CORES, run_timeout=RUN_TIMEOUT, max_attempts=MAX_ATTEMPTS)
    campaign.wait_for(WAIT_FOR)
    campaign.hold_lock()
    try:
        campaign.provenance(dt=DT, steps=STEPS, seed=SEED)
        levels = mesh_levels()
        rng = random.Random(SEED)
        failures = 0
        for stage in STAGES:
            selected = select_presets(STUDY)
            runs = stage_specs(stage, selected, campaign.records())
            rng.shuffle(runs)
            log(f"stage {stage}: {len(runs)} runs")
            for run in runs:
                while not campaign.finished(run["run_id"]):
                    run, level = resolve_level(run, levels)
                    if level is None:
                        log(f"no mesh level for {run['run_id']}; skipped")
                        campaign.append(dict(run, status="no-mesh", attempt=1))
                        continue
                    row = campaign.execute(run, command(run, level), level, foreign_test=run["foreign"])
                    failures = failures + 1 if row["status"] == "failed" else 0
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        log(f"{failures} consecutive failed runs: stopping (see runs/logs); restart resumes")
                        return
        select_presets(STUDY)
        smoke_report(STUDY)
        MARKER.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
        log("D8-S done")
    finally:
        campaign.release()


if __name__ == "__main__":
    main()
