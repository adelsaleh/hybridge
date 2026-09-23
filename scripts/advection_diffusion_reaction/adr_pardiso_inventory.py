"""Read-only inventory/comparison of the archived ADR report matrices.

Uses the completed pMG-AMG coverage manifest instead of regenerating meshes or
coefficients. Report occurrences remain aliases of one matrix/RHS identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "run_outputs/solver_studies/adr_scaling_2026_09_17"
NATIVE = STUDY / "native_hp_completion_2026_09_21"
STRESS = ROOT / "run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k"
CACHE_FILES = ("system_blocks.npy", "system_neighbors.npy", "system_rhs.npy")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def result_rows(data):
    """Descend report containers only, never samples, warmups or manifests."""
    if isinstance(data, list):
        for entry in data:
            yield from result_rows(entry)
    elif isinstance(data, dict):
        if "candidate" in data and "status" in data:
            yield data
        else:
            for key in ("rows", "campaigns"):
                yield from result_rows(data.get(key, []))


def timing_metrics(row):
    """Only full, accepted measurements receive ranked times; never warmups."""
    samples = row.get("samples", [])
    if row.get("status") != "passed" or row.get("profile_note") or not samples:
        return None
    if len(samples) != row.get("requested_repeats", len(samples)):
        return None
    trials = samples + row.get("warmups", [])
    if any(len(t.get("solves", [])) != 2 for t in trials):
        return None
    solves = [s for t in trials for s in t["solves"]]
    if any(not s.get("passed") or not math.isfinite(s.get("true_relative_residual", math.inf))
           or s["true_relative_residual"] > row.get("rtol", 1e-10) for s in solves):
        return None
    times = [s.get("solve_ms", math.nan) for s in solves] + [t.get("setup_ms", math.nan) for t in trials]
    if any(not math.isfinite(t) or t < 0 for t in times):
        return None
    return dict(setup_median_ms=statistics.median(t["setup_ms"] for t in samples),
                fresh_median_ms=statistics.median(t["setup_ms"] + t["solves"][0]["solve_ms"] for t in samples),
                reused_mean_ms=statistics.mean(t["solves"][1]["solve_ms"] for t in samples),
                fresh_min_ms=min(t["setup_ms"] + t["solves"][0]["solve_ms"] for t in samples),
                fresh_max_ms=max(t["setup_ms"] + t["solves"][0]["solve_ms"] for t in samples),
                reused_min_ms=min(t["solves"][1]["solve_ms"] for t in samples),
                reused_max_ms=max(t["solves"][1]["solve_ms"] for t in samples),
                measured_setups=len(samples), worst_relative_residual=max(s["true_relative_residual"] for s in solves))


def family(row):
    candidate = row["candidate"].lower()
    if candidate.startswith("native_hp"):
        return "pMG-AMG"
    if candidate.startswith("asm"):
        return "ASM+PP"
    if candidate.startswith("bj"):
        return "BJ+PP"
    if candidate.startswith("pp"):
        return "PP"
    # AMGX is a library, not synonymous with an AMG hierarchy. Preserve the
    # complete configuration below even when using these coarse family labels.
    return "AMGX" if row.get("family") == "amgx" or candidate.startswith(("amgx", "amg")) else row.get("family", candidate)


def annotate_levels(systems):
    ladders = defaultdict(set)
    for system in systems:
        for origin in system["origins"]:
            if origin["suite"] in ("smooth", "oscillatory", "focused_scaling") and origin["p"] == 6:
                ladders[(origin["suite"], origin["campaign"], origin["case"], origin["geometry"])].add(origin["triangles"])
    for system in systems:
        for origin in system["origins"]:
            key = (origin["suite"], origin["campaign"], origin["case"], origin["geometry"])
            if origin["p"] == 6 and key in ladders:
                origin["mesh_level"] = "L" + str(sorted(ladders[key]).index(origin["triangles"]) + 1)
            if origin["suite"] in ("smooth", "oscillatory", "focused_scaling") and origin.get("nominal_n") == 64:
                origin["degree_sweep"] = True


def inventory(native_campaign=NATIVE, stress_campaign=STRESS):
    """Return all earlier report systems plus every completed annular stress mesh.

    Includes the original exploratory controls from main.tex as well as the
    synthesis: deliberately broader than just the winning curves. No square
    stress results or in-progress campaigns are read by default.
    """
    import numpy as np

    native_campaign, stress_campaign = Path(native_campaign), Path(stress_campaign)
    manifest_path = native_campaign / "manifest.json"
    manifest = read(manifest_path)
    systems = {s["system_id"]: {k: v for k, v in s.items() if k != "existing"}
               for s in manifest["systems"]}
    if len(systems) != len(manifest["systems"]):
        raise ValueError("Duplicate systems in completed coverage manifest")
    sources = {manifest_path, native_campaign / "all_native_results.json"}
    sources.update(Path(o["source"]) for s in systems.values() for o in s["origins"])
    # Retain published tuned endpoints and stronger hierarchy probes too.
    sources.update(p for p in (STUDY / "oscillatory/asm_optimality_audit").glob("*_raw_final.json"))
    sources.update(Path(p) for p in manifest.get("source_artifacts", {})
                   if "/strong_probe/" in p and p.endswith(".json") and not p.endswith(".spec.json"))
    stress_summary = stress_campaign / "summary.json"
    sources.add(stress_summary)
    stress_rows = list(result_rows(read(stress_summary)))
    groups = defaultdict(list)
    for row in stress_rows:
        groups[(row["case"], row["target_triangles"], row["p"])].append(row)
    for (case, target, p), rows in groups.items():
        hashes = {r.get("operator_sha256") for r in rows}
        if len(hashes) != 1 or None in hashes:
            raise ValueError(f"Missing or inconsistent stress identity: {case}/{target}/p{p}")
        identity = hashes.pop()
        spec_path = stress_campaign / f"specs/assemble_{case}_t{target}_p{p}.json"
        spec = read(spec_path)
        sources.add(spec_path)
        representative = rows[0]
        origin = dict(suite="stress", campaign=stress_campaign.name, source=str(stress_summary),
                      case=case, geometry=spec.get("stress_parameters", {}).get("geometry", "annulus"),
                      p=p, target_triangles=target, triangles=representative["triangles"],
                      trace_dofs=representative["trace_dofs"])
        if identity in systems:
            systems[identity]["origins"].append(origin)
        else:
            systems[identity] = dict(system_id=identity, case=case, p=p, cache=spec["cache"],
                                    geometry=origin["geometry"], triangles=origin["triangles"],
                                    trace_dofs=origin["trace_dofs"], origins=[origin])
    if not systems or not groups:
        raise ValueError("Missing report or stress systems; refusing an incomplete campaign")
    by_cache = {}
    for system in systems.values():
        cache = Path(system["cache"]).resolve(strict=True)
        system["cache"] = str(cache)
        by_cache[str(cache)] = system["system_id"]
        system["cache_files"] = {name: dict(size=(cache/name).stat().st_size,
                                           mtime_ns=(cache/name).stat().st_mtime_ns) for name in CACHE_FILES}
        blocks, neighbors, rhs = [np.load(cache / name, mmap_mode="r", allow_pickle=False) for name in CACHE_FILES]
        if blocks.ndim != 4 or blocks.shape[2] != blocks.shape[3] or neighbors.shape != blocks.shape[:2]:
            raise ValueError(f"Invalid face layout: {cache}")
        if rhs.size != system["trace_dofs"] or rhs.size != blocks.shape[0] * blocks.shape[2]:
            raise ValueError(f"Archived DOF count does not match cache: {cache}")
        # Hash small topology file now; full floating-point hash is checked in
        # each isolated worker, outside the timing interval.
        system["neighbors_file_sha256"] = digest(cache / "system_neighbors.npy")
        system["face_storage_bytes"] = int(blocks.nbytes + neighbors.nbytes + rhs.nbytes)
        system["key"] = f"{system['geometry']}_{system['case']}_p{system['p']}_{system['system_id'][:12]}"
        system["comparators"] = []
    for path in sorted(sources):
        for row in result_rows(read(path)):
            identity = row.get("operator_sha256")
            if not identity and row.get("matrix_cache"):
                identity = by_cache.get(str(Path(row["matrix_cache"]).resolve()))
            if identity not in systems:
                continue
            systems[identity]["comparators"].append(dict(
                source=str(path.resolve()), candidate=row["candidate"], family=family(row),
                status=row["status"], metrics=timing_metrics(row),
                configuration=row.get("configuration", {}), policy=row.get("policy"),
                rtol=row.get("rtol"), operator_sha256=identity,
                identity_match="operator_hash" if row.get("operator_sha256") else "cache_path",
                peak_process_rss_kib=row.get("peak_process_rss_kib")))
    entries = sorted(systems.values(), key=lambda s: (s["trace_dofs"], s["key"]))
    annotate_levels(entries)
    if any(not s["comparators"] for s in entries):
        raise ValueError("A system has no archived iterative result")
    return entries, {str(p.resolve()): digest(p) for p in sorted(sources)}


def coverage(systems):
    return dict(unique_systems=len(systems), max_trace_dofs=max(s["trace_dofs"] for s in systems),
                systems_by_suite=dict(Counter(o for s in systems for o in {v["suite"] for v in s["origins"]})),
                mesh_levels=sorted({o["mesh_level"] for s in systems for o in s["origins"] if "mesh_level" in o}),
                degrees=sorted({s["p"] for s in systems}))


def comparisons(system, direct):
    """Pair every archived configuration with the same direct system, not DOFs."""
    if direct.get("measurement_phase") == "tuning":
        return []  # Pilots must never become the published CPU baseline.
    direct_metrics = timing_metrics(direct)
    selected_for = direct.get("selected_for", ["fresh", "reused"])
    rows = []
    for comparator in system["comparators"]:
        item = dict(system_id=system["system_id"], case=system["case"], geometry=system["geometry"],
                    p=system["p"], triangles=system["triangles"], trace_dofs=system["trace_dofs"],
                    threads=direct.get("threads"), direct_status=direct.get("status"),
                    measurement_phase=direct.get("measurement_phase", "measurement"),
                    selected_for=selected_for, selection_fingerprint=direct.get("selection_fingerprint"), **comparator)
        item["direct_metrics"] = direct_metrics
        item["speedup_fresh"] = item["speedup_reused"] = None
        if direct_metrics and comparator["metrics"]:
            for label, key in (("fresh", "fresh_median_ms"), ("reused", "reused_mean_ms")):
                if label in selected_for and direct_metrics[key] > 0:
                    item["speedup_" + label] = comparator["metrics"][key] / direct_metrics[key]
        rows.append(item)
    return rows
