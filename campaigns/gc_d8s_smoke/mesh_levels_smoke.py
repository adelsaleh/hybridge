"""D8-S meshes: scale s per (family, level, p) so that triangles * (p+1)(p+2)/2 is within 2% of the target.

Meshes are built with the diocotron driver's own ``_run_mesh`` (gmsh mesh cache on), from the band and
outer sizes written as strings, so the smoke runs that pass the same strings hit the cache.
Mesh generation only (no solve). Usage: mesh_levels_smoke.py OUT.jsonl family:level:p [...]
family: m128 (hb = 0.002 s, h = 0.04 s) or zg (hb = 0.008 s, h = 0.04 s).
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import json, math, sys, time
from pathlib import Path
from scripts.guiding_center.studies.diocotron.run_diocotron_benchmark import _benchmark_cases, _run_mesh

TARGETS = {"m128": {"L2": 3.0e6, "L3": 9.0e6, "L4": 2.7e7}, "zg": {"L3": 3.0e6, "L4": 9.0e6}}
BASE = {"m128": (0.002, 0.04), "zg": (0.008, 0.04)}
CASES = {"m128": next(_benchmark_cases("disk-highmode", modes=(128,), epsilon=None))[1],
         "zg": next(_benchmark_cases("disk-zg", modes=(), epsilon=None))[1]}
# Triangles at s = 1 and the exponent of N ~ s^-alpha, from the 2026-10-16 calibrations.
GUESS = {"m128": (103500, 1.84), "zg": (13650, 1.87)}


def sizes(family, s):
    hb, h = BASE[family]
    return f"{hb * s:.7g}", f"{h * s:.7g}"


def build(family, s):
    hb, h = sizes(family, s)
    start = time.perf_counter()
    mesh = _run_mesh(CASES[family], float(hb), float(h))
    return hb, h, int(mesh.num_tri), int(mesh.num_edg), time.perf_counter() - start


def main():
    out = Path(sys.argv[1])
    done = {(r["family"], r["level"], r["p"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    for spec in sys.argv[2:]:
        family, level, p = spec.split(":"); p = int(p)
        if (family, level, p) in done:
            continue
        n_p = (p + 1) * (p + 2) // 2
        target = TARGETS[family][level] / n_p
        n1, alpha = GUESS[family]
        s = (n1 / target) ** (1 / alpha)
        for attempt in range(8):
            for retry in range(3):  # gmsh occasionally returns an invalid mesh at a given size; perturb it
                try:
                    hb, h, tri, edg, seconds = build(family, s)
                    break
                except ValueError as error:
                    print(f"{family} {level} p={p} s={s:.5f}: {error}; retrying with s*1.003", flush=True)
                    s *= 1.003
            else:
                raise RuntimeError(f"no valid mesh for {family} {level} p={p}")
            print(f"{family} {level} p={p} s={s:.5f} hb={hb} h={h} triangles={tri} ({tri / target - 1:+.3%}) {seconds:.1f}s",
                  flush=True)
            if abs(tri / target - 1) < 0.02:
                break
            s *= (tri / target) ** (1 / alpha)
        row = dict(family=family, level=level, p=p, s=round(s, 6), hb=hb, h=h, triangles=tri, edges=edg,
                   element_unknowns=tri * n_p, trace_unknowns=edg * (p + 1), target=TARGETS[family][level],
                   attempts=attempt + 1)
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
