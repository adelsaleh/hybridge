#!/usr/bin/env python3
"""D8: is high order worth it for the diocotron? Performance campaign at equal element unknowns, p = 5..9.

Design: PLAN.md, D8 (user decisions of 2026-10-16: p = 5-9, T = 70, SI-BDF2 first, C1 = the m = 128
Gaussian ring, C2 = the Zoni-Guclu truncated ring; one T = 200 run at the best (p, h) comes after this
campaign). Protocol: campaigns/common/protocol.py (campaign-wide GPU lock, idle
GPU before each run, nvidia-smi samples, contamination requeue, host process groups of pause_pgids.json
paused during runs, provenance, resume skips finished runs). Each run is one process of the diocotron driver
(scripts/guiding_center/studies/diocotron/run_diocotron_benchmark.py) with synchronized step timing, the enstrophy
budget, diagnostics every DIAGNOSTICS_EVERY steps, movies off and full snapshots (rho_h, phi_h, q_h) at the
observation times; its per-step JSONL and summary rows are kept.

Phases, in this order (within a phase and level the degrees run in a seeded random order):
  I   intermediate campaign (user, 2026-10-17 evening): p = 5 at L3 (9 M element unknowns) against p = 9 at L2
      (3 M; INTERMEDIATE), dt = 0.0125, T = 70, both no-KKT runs first, then each with the KKT positivity
      projection (run ids ..._kkt), diagnostics every 5 steps, verbosity 3; which degree is faster and better
      with SI-BDF2 before the full ladder                                                              4 runs
  X   long run (user, 2026-10-19): p = 9 at L2 to T = 400, no KKT, snapshots at SNAPSHOT_TIMES_LONG, verbosity 2  1 run
  P2a spatial errors, C1 at L0, L1, L2 (dt = 0.0125, T = 70)                          15 runs
  P1  temporal constant, C1 at L2, p = 5 and 8, dt = 0.1, 0.05, 0.025, 0.00625 (0.0125 from P2a)   8 runs
  P2b spatial errors, C1 at L3, then C2 at L0..L3                                        25 runs
  R   references at L4: C1 (27 M element unknowns) and C2 (9 M), p = 5 and 8              4 runs
  T   re-timing, C1 at L2, p = 5, 7, 9, 500 steps (run-to-run spread)                    3 runs
  P3  cost at a given error: the runs listed in phase3_plan.jsonl (family, level, p, dt), written after
      the P1/P2 analysis and the decision point with the user; read again before every run
A run is identified by (family, level, p, dt[, tag][, kkt]); P1 and P2 share their dt = 0.0125 runs and
the no-KKT runs of phase I are the L3 p = 5 run of P2b and the L2 p = 9 run of P2a.
Presets per degree come from the smoke campaign (../diocotron_order_smoke_20261016/presets_selected.json:
Poisson and transport candidates) and VQ below (volume rule from its stage E); the drift from DRIFT below.

Start detached (see this campaign README; originally from the output directory; --phases I,P2a restricts the launch to the listed phases):
  setsid nohup taskset -c 12-23 ./launcher.py [--phases I] > driver_stdout.log 2>&1 < /dev/null &
Stop: kill -- -<pgid>. Resume: start it again. Plan without running: ./launcher.py [--phases I] --plan.
Completion marker: D8_DONE for all phases, D8_<phases>_DONE (e.g. D8_I_DONE) for a restricted launch.
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

STUDY = output_directory('diocotron_order_20261016')
SMOKE = STUDY.parent / "diocotron_order_smoke_20261016"
HYBRIDGE = REPO
sys.path.insert(0, str(HYBRIDGE))
from campaigns.common.protocol import PYTHON, Campaign, log  # noqa: E402

RUNS = STUDY / "runs"
MARKER = STUDY / "D8_DONE"
PHASE3_PLAN = STUDY / "phase3_plan.jsonl"
DEGREES = (5, 6, 7, 8, 9)
LEVELS = ("L0", "L1", "L2", "L3")
FINAL_TIME = 70.0
DT_S = 0.0125  # Phase 2 time step (PLAN D8: the time error about 5 % of the spatial error)
DT_LADDER = (0.1, 0.05, 0.025, 0.0125, 0.00625)
P1_DEGREES = (5, 8)
REFERENCE_DEGREES = (5, 8)
RETIME = dict(degrees=(5, 7, 9), steps=500)
SNAPSHOT_TIMES = {"m128": (30, 40, 45, 50, 55, 60, 70), "zg": (30, 40, 50, 60, 70)}
SNAPSHOT_TIMES_LONG = (30, 40, 45, 50, 55, 60, 70, 80, 100, 120, 160, 200, 240, 300, 400)  # phase X
LONG = dict(level="L2", p=9, final_time=400.0)
TAU = {"m128": 128000.0, "zg": 32000.0}
BENCHMARK = {"m128": ["--benchmark", "disk-highmode", "--modes", "128"], "zg": ["--benchmark", "disk-zg"]}
# Drift (PLAN D8, decided 2026-10-16 night): V2, the compatible drift from the p+1 postprocessed potential
# (velocity "compatible", offset 0; "--compatible-potential postprocessed", the path the convergence
# session validated). Basis: on the device, Gmsh p = 2-6 (IMEX-ARK3, bsr), V2 matches the raw drift with
# Poisson degree p+1 on every mesh (interior rho L2 ratio 0.96-1.03, same orders) and costs 8-37 % less
# per step at p = 2-9 (SI-BDF2 timing); p = 9 has no Poisson degree 10, so V2 is the one drift of
# degree p available at every p of the campaign. Alternatives kept for reruns: raw q = p
# (dict(velocity="raw", offset=0)) and raw q = p+1 (offset 1; p <= 8 only).
DRIFT = dict(velocity="compatible", offset=0)
# 1-D Gauss count of the collapsed volume rule per degree: (3p+2)//2 makes the rule exact for the advection
# term with a degree-p drift at every p (the default Dunavant 2p rule under-integrates it at p <= 7; at
# p >= 8 the default (2p+2)^2 rule is exact but the diagnostics then cost twice as much). Smoke stage E
# measures the cost of this policy; None keeps the driver's default.
VQ: dict[int, int | None] = {p: (3 * p + 2) // 2 for p in DEGREES}
# Poisson stopping rule (smoke stage J, 2026-10-17): the reduced right-hand side of these systems has norm
# 4-8e-5, so the driver's default absolute 1e-10 is only about 1e-6 relative and two Poisson solvers then
# differ by 1.5e-7 in the density; at 1e-14 (relative 1e-10, reached) they agree to 2e-11, for +3-40 % step.
POISSON_ATOL = 1e-14
DIAGNOSTICS_EVERY = {"I": 5, "X": 5}  # steps between diagnostics rows per phase; DEFAULT_DIAGNOSTICS_EVERY elsewhere
DEFAULT_DIAGNOSTICS_EVERY = 10
VERBOSITY = {"I": 3}  # driver verbosity per phase (3: every timing, solver and diagnostic line); 2 elsewhere
DEFAULT_VERBOSITY = 2
EPSILON = 1e-4  # seed amplitude of the disk-highmode case (the driver's default, explicit for the provenance)
KKT_POINTS = "quadrature+lattice"  # enforcement points of the KKT positivity projection (phase I _kkt runs)
# Intermediate campaign meshes (user, 2026-10-17 17:10: "for the p=9 we should take less triangles"): p = 5 at L3
# (9 M element unknowns, 428,618 triangles) against p = 9 at L2 (3 M, about 55k triangles, the largest h of the ladder
# that still resolves the ring: 1.7 band elements across the ring width, 17 unknowns across it at p = 9).
# 2026-10-18 (user: "try a p8 run with 75k triangles", fastest configs): p = 8 at the custom level T75k (74,710 triangles,
# 3.4 M element unknowns; row in mesh_levels_m128.jsonl), without KKT only.
INTERMEDIATE = dict(levels={5: "L3", 9: "L2", 8: "T75k"}, kkt_degrees=(5, 9))
SEED = 20261017
MAX_ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 3
RUN_TIMEOUT = 8 * 3600.0  # the L4 references take about 3 h, the T = 400 run about 4 h
CPU_CORES = "12-23"


def slug(text: str) -> str:
    return text.replace("+", "-").replace(".", "p")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def mesh_levels() -> dict:
    """(family, level, p) → mesh row: C1 L0-L3 from mesh_levels_m128.jsonl, C2 L0-L3 from mesh_levels.jsonl
    (family zg), both L4 from the smoke campaign's table."""
    rows = {}
    for row in _jsonl(record_file(STUDY, CODE, "mesh_levels_m128.jsonl")):
        if "triangles" in row and "level" in row:
            rows[("m128", row["level"], row["p"])] = row
    for row in _jsonl(record_file(STUDY, CODE, "mesh_levels.jsonl")):
        if row.get("family") == "zg":
            rows[("zg", row["level"], row["p"])] = row
    for row in _jsonl(record_file(SMOKE, CODE.parent / "gc_d8s_smoke", "mesh_levels_smoke.jsonl")):
        if row["level"] == "L4":
            rows[(row["family"], "L4", row["p"])] = row
    return rows


def presets() -> dict:
    path = record_file(SMOKE, CODE.parent / "gc_d8s_smoke", "presets_selected.json")
    return {int(p): choice for p, choice in json.loads(path.read_text()).items()} if path.exists() else {}


def spec(phase, family, level, p, dt, *, steps=None, snapshots=True, tag=None, kkt=False, final_time=None):
    final_time = (FINAL_TIME if steps is None else steps * dt) if final_time is None else final_time
    run_id = f"{family}{level}_p{p}_dt{slug(f'{dt:g}')}" + (f"_{tag}" if tag else "") + ("_kkt" if kkt else "")
    return dict(run_id=run_id, phase=phase, family=family, level=level, p=p, q=p + DRIFT["offset"], dt=dt,
                final_time=final_time, steps=int(round(final_time / dt)), snapshots=snapshots,
                velocity=DRIFT["velocity"], offset=DRIFT["offset"], vq=VQ.get(p), kkt=kkt)


def phase_specs(phase: str, rng: random.Random | None = None) -> list[dict]:
    """Run specifications of one phase, degrees shuffled within each (family, level) when rng is given."""
    def shuffled(groups):
        out = []
        for group in groups:
            if rng is not None:
                rng.shuffle(group)
            out += group
        return out

    if phase == "I":
        # Order (user, 2026-10-17 17:05): both no-KKT runs first (the p = 5/p = 9 timing answer), then the KKT runs.
        return [spec("I", "m128", level, p, DT_S, kkt=kkt)
                for kkt in (False, True) for p, level in INTERMEDIATE["levels"].items()
                if not kkt or p in INTERMEDIATE["kkt_degrees"]]
    if phase == "X":
        return [spec("X", "m128", LONG["level"], LONG["p"], DT_S, final_time=LONG["final_time"],
                     tag=f"T{LONG['final_time']:g}")]
    if phase == "P2a":
        return shuffled([[spec("P2a", "m128", level, p, DT_S) for p in DEGREES] for level in ("L0", "L1", "L2")])
    if phase == "P1":
        return shuffled([[spec("P1", "m128", "L2", p, dt) for dt in DT_LADDER if dt != DT_S] for p in P1_DEGREES])
    if phase == "P2b":
        groups = [[spec("P2b", "m128", "L3", p, DT_S) for p in DEGREES]]
        groups += [[spec("P2b", "zg", level, p, DT_S) for p in DEGREES] for level in LEVELS]
        return shuffled(groups)
    if phase == "R":
        return [spec("R", family, "L4", p, DT_S) for family in ("m128", "zg") for p in REFERENCE_DEGREES]
    if phase == "T":
        return [spec("T", "m128", "L2", p, DT_S, steps=RETIME["steps"], snapshots=False, tag="retime")
                for p in RETIME["degrees"]]
    if phase == "P3":
        return [spec("P3", row["family"], row["level"], int(row["p"]), float(row["dt"]), tag=row.get("tag"))
                for row in _jsonl(PHASE3_PLAN)]
    raise ValueError(phase)


PHASES = ("I", "X", "P2a", "P1", "P2b", "R", "T", "P3")


def command(run: dict, level: dict, preset: dict) -> list[str]:
    args = [str(PYTHON), "-u", "-m", "scripts.guiding_center.studies.diocotron.run_diocotron_benchmark",
            *BENCHMARK[run["family"]], "--orders", str(run["p"]), "--band-mesh-sizes", str(level["hb"]),
            "--mesh-size", str(level["h"]), "--dts", f"{run['dt']:g}", "--final-time", f"{run['final_time']:g}",
            "--schemes", "si-bdf2", "--velocities", run["velocity"], "--poisson-tau", f"{TAU[run['family']]:g}",
            "--poisson-candidate", preset["poisson"], "--transport-candidate", preset["transport"],
            "--poisson-order-offset", str(run["offset"]), "--synchronize-step-timing", "--enstrophy-budget",
            "--movie-interval", "0", "--epsilon", f"{EPSILON:g}",
            "--diagnostics-every", str(DIAGNOSTICS_EVERY.get(run["phase"], DEFAULT_DIAGNOSTICS_EVERY)),
            "--output-dir", str(RUNS), "--prefix", "d8", "--run-tag", run["run_id"],
            "--verbosity", str(VERBOSITY.get(run["phase"], DEFAULT_VERBOSITY))]
    if run["snapshots"]:
        times = SNAPSHOT_TIMES_LONG if run["phase"] == "X" else SNAPSHOT_TIMES[run["family"]]
        args += ["--snapshot-times", ",".join(f"{t:g}" for t in times if t <= run["final_time"])]
    if run["velocity"] == "compatible":
        args += ["--compatible-potential", "postprocessed"]
    if run.get("vq"):
        args += ["--volume-quad-1d", str(run["vq"])]
    args += ["--poisson-solver-atol", f"{POISSON_ATOL:g}"]
    if run.get("kkt"):
        args += ["--density-positivity", "kkt", "--density-positivity-points", KKT_POINTS]
    return args


def plan(phases) -> None:
    """Print the selected phases' runs with their meshes and presets (no lock, no GPU)."""
    levels, chosen = mesh_levels(), presets()
    total = 0
    for phase in phases:
        runs = phase_specs(phase)
        total += len(runs)
        print(f"phase {phase}: {len(runs)} runs")
        for run in runs:
            level = levels.get((run["family"], run["level"], run["p"]))
            preset = chosen.get(run["p"])
            print("  ", run["run_id"], f"{run['steps']} steps",
                  "NO MESH" if level is None else f"{level['triangles']} triangles",
                  "NO PRESET" if preset is None else f"{preset['poisson']} / {preset['transport']}")
    print(f"{total} runs")
    for run in phase_specs(phases[0]):
        level, preset = levels.get((run["family"], run["level"], run["p"])), chosen.get(run["p"])
        if level is not None and preset is not None:
            print(" ".join(command(run, level, preset)))


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phases", default=",".join(PHASES),
                        help=f"comma-separated phases to run, in campaign order (default all: {','.join(PHASES)})")
    parser.add_argument("--plan", action="store_true", help="print the runs and one command per phase, no run")
    args = parser.parse_args(argv)
    chosen = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    unknown = [phase for phase in chosen if phase not in PHASES]
    if unknown:
        parser.error(f"unknown phases {unknown}; known: {PHASES}")
    args.phases = tuple(phase for phase in PHASES if phase in chosen)
    return args


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.plan:
        plan(args.phases)
        return
    marker = MARKER if args.phases == PHASES else STUDY / f"D8_{'_'.join(args.phases)}_DONE"
    os.chdir(STUDY)
    RUNS.mkdir(exist_ok=True)
    campaign = Campaign(STUDY, cpu_cores=CPU_CORES, run_timeout=RUN_TIMEOUT, max_attempts=MAX_ATTEMPTS)
    campaign.hold_lock()
    try:
        campaign.provenance(final_time=FINAL_TIME, dt_s=DT_S, drift=DRIFT, vq=VQ, poisson_atol=POISSON_ATOL, seed=SEED,
                            presets=presets(), phases=list(args.phases), epsilon=EPSILON, kkt_points=KKT_POINTS,
                            diagnostics_every=dict(DIAGNOSTICS_EVERY, default=DEFAULT_DIAGNOSTICS_EVERY),
                            verbosity=dict(VERBOSITY, default=DEFAULT_VERBOSITY))
        levels = mesh_levels()
        rng = random.Random(SEED)
        failures = 0
        for phase in args.phases:
            runs = phase_specs(phase, rng)
            log(f"phase {phase}: {len(runs)} runs")
            for run in runs:
                while not campaign.finished(run["run_id"]):
                    level, preset = levels.get((run["family"], run["level"], run["p"])), presets().get(run["p"])
                    if level is None or preset is None:
                        log(f"{run['run_id']}: {'no mesh level' if level is None else 'no preset'}; skipped")
                        campaign.append(dict(run, status="no-mesh" if level is None else "no-preset", attempt=1))
                        continue
                    row = campaign.execute(dict(run, poisson=preset["poisson"], transport=preset["transport"]),
                                           command(run, level, preset), level)
                    failures = failures + 1 if row["status"] == "failed" else 0
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        log(f"{failures} consecutive failed runs: stopping (see runs/logs); restart resumes")
                        return
        marker.write_text(datetime.now().isoformat(timespec="seconds") + "\n")
        log(f"D8 done ({marker.name})")
    finally:
        campaign.release()


if __name__ == "__main__":
    main()
