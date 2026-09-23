"""Replay saved hybrid systems, export their actual hierarchy, and test BSR.

No mesh generation, assembly, time integration, builds, or JIT compilation.
Run the user-owned AMGX rebuild before --phase all/measure. --phase baseline
can run with the existing library. All outputs are checkpointed sequentially.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from hdgfem.linalg.hierarchy_bsr import (load_operator, level_permutation,
    permute_operator, padded_bsr, verify_reconstruction, storage_stats)
from scripts.guiding_center.poisson.amgx_bsr_smoothing import ROOT
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.poisson.hierarchy_product_benchmark import benchmark_operator

SYSTEMS = {
    157280: ROOT/'artifacts/full_bsr_convergence_20260914/p6_150k_screen1',
    315425: ROOT/'artifacts/full_bsr_convergence_20260914/p6_300k_controls',
}


def save(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay(systems, output, *, diagnostic=False):
    if output.exists():
        result = json.loads((output/'completion.json').read_text())
        if result['status'] != 'complete' or result.get('rejected'):
            raise RuntimeError(f'Incomplete replay at {output}; inspect it before using a fresh output')
        metadata = json.loads((output/'metadata.json').read_text())
        if Path(metadata['source_systems']) != systems.resolve() or metadata.get('diagnostic_profile', False) != diagnostic:
            raise ValueError('Existing replay provenance does not match the requested run')
        return
    command = [sys.executable, '-m', 'scripts.guiding_center.poisson.replay_poisson_bsr',
        '--systems-dir', str(systems), '--output-dir', str(output),
        '--candidates', 'hybrid_l1_0_3', '--repetitions', '1' if diagnostic else '6']
    if diagnostic:
        command += ['--hierarchy-export-prefix', str(output/'hierarchy'/'hybrid'), '--profile-hierarchy']
    subprocess.run(command, cwd=ROOT, check=True)


def profile_data(folder):
    result = {}
    for path in sorted(folder.glob('profile_repeat0_step*.csv')):
        step = int(path.stem.split('step')[-1])
        per_step = {}
        with path.open() as stream:
            for key, ms in csv.reader(stream):
                per_step.setdefault(key, []).append(float(ms))
        if not per_step:
            raise ValueError(f'No native product records in {path}')
        result[step] = per_step
    if len(result) != 3:
        raise ValueError('Require product profiles for all three saved systems')
    return result


def verify_replays(folder):
    baseline = rows(folder/'baseline/poisson_samples.jsonl')
    diagnostic = rows(folder/'diagnostic/poisson_samples.jsonl')
    for d in diagnostic:
        matches = [r for r in baseline if r['step'] == d['step']]
        if len(matches) != 6:
            raise ValueError('Expected a warmup and five normal solves for each saved system')
        for r in matches:
            for key in ('iterations', 'rhs_sha256', 'initial_guess_sha256', 'status'):
                if r[key] != d[key]:
                    raise ValueError(f'Export/profile replay changed {key} for step {d["step"]}')
            if not np.allclose(r['residual_history'], d['residual_history'], rtol=1e-12, atol=0.):
                raise ValueError('Export/profile replay changed residual history')
    return baseline


def measure(folder):
    baseline = verify_replays(folder)
    profiles = profile_data(folder/'diagnostic')
    exports = sorted((folder/'diagnostic/hierarchy').glob('hybrid.L*.json'))
    if not exports:
        raise RuntimeError('No native exports. Rebuild AMGX using the documented command.')
    metadata = {p: json.loads(p.read_text()) for p in exports}
    a_paths = {m['level']: p for p, m in metadata.items() if m['role'] == 'A'}
    levels = sorted(a_paths)
    if not levels or levels != list(range(1, levels[-1]+1)):
        raise ValueError('Incomplete coarse hierarchy')
    expected = {(level, 'A') for level in levels}
    expected |= {(level, role) for level in range(levels[-1]) for role in ('P', 'R')}
    if {(m['level'], m['role']) for m in metadata.values()} != expected:
        raise ValueError('Incomplete A/P/R exports')
    result_dir = folder/'products'
    result_dir.mkdir(exist_ok=True)
    permutation_seconds = {}
    # Only permutations are retained; each coarse matrix is released before the next.
    for ordering in ('original', 'rcm'):
        started = time.perf_counter()
        for level, path in a_paths.items():
            a, _ = load_operator(path)
            np.save(result_dir/f'{ordering}.L{level}.permutation.npy', level_permutation(a, ordering))
            del a
        permutation_seconds[ordering] = time.perf_counter()-started
    records = []
    with kernel_cache_only(True):
        for path in exports:
            a, meta = load_operator(path)
            level, role = meta['level'], meta['role']
            rowlevel, collevel = ((level, level) if role == 'A' else
                (level, level+1) if role == 'P' else (level+1, level))
            key = f'L{level}.{role}'
            for ordering in ('original', 'rcm'):
                def permutation(lev, size):
                    return (np.arange(size, dtype=np.int32) if lev == 0 else
                            np.load(result_dir/f'{ordering}.L{lev}.permutation.npy'))
                rp, cp = permutation(rowlevel, a.shape[0]), permutation(collevel, a.shape[1])
                started = time.perf_counter()
                permuted = permute_operator(a, rp, cp)
                permute_seconds = time.perf_counter()-started
                for block in (2, 4, 7, 8):
                    output = result_dir/f'{key}.{ordering}.b{block}.json'
                    # Product measurements are rerun as a unit after interruption;
                    # never combine timing batches taken in different sessions.
                    started = time.perf_counter()
                    bsr = padded_bsr(permuted, block)
                    conversion_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    verify_reconstruction(a, bsr, rp, cp)
                    check_seconds = time.perf_counter()-started
                    record = dict(operator=key, level=level, role=role, ordering=ordering,
                        block_size=block, shape=list(a.shape), storage=storage_stats(permuted, bsr),
                        export_bytes=meta['export_bytes'], permutation_bytes=rp.nbytes+cp.nbytes,
                        permutation_seconds=permute_seconds, conversion_seconds=conversion_seconds,
                        reconstruction_seconds=check_seconds, exact_reconstruction=True,
                        **benchmark_operator(permuted, bsr))
                    save(output, record); records.append(record)
                    print(f'[product] {folder.name} {key} {ordering} b={block}', flush=True)
                    del bsr
                del permuted
            del a
    save(folder/'comparison.json', dict(records=records, permutation_seconds=permutation_seconds))
    write_report(folder, records, baseline, profiles, permutation_seconds)


def write_report(folder, records, baseline, profiles, permutation_seconds):
    """Project only observed multiply() work; hold all other work constant."""
    keys = sorted({r['operator'] for r in records})
    for key in set().union(*(set(p) for p in profiles.values())) - {'L0.A'}:
        if key not in keys:
            raise ValueError(f'Profile operator {key} has no exported matrix')
    # Each of three saved RHS has equal weight; discard the first replay round.
    solves = {s: [r['solve_seconds']*1000 for r in baseline
                  if r['step'] == s and r['repetition'] > 0] for s in profiles}
    normal = statistics.mean(statistics.median(v) for v in solves.values())
    normal_spread = statistics.mean(max(v)-min(v) for v in solves.values())
    counts = {key: statistics.mean(len(p.get(key, [])) for p in profiles.values()) for key in keys}
    replay_metadata = json.loads((folder/'baseline/metadata.json').read_text())
    source = Path(replay_metadata['source_systems'])
    source_metadata = json.loads((source/'metadata.json').read_text())
    fine_bytes = int(source_metadata['matrix_bytes'])
    source_samples = rows(source/'poisson_samples.jsonl')
    captured = {r['step']: r for r in source_samples if r.get('variant') == 'hybrid_l1_0_3'}
    baseline_checks = []
    for step in profiles:
        current = [r for r in baseline if r['step'] == step and r['repetition'] > 0]
        saved = captured[step]
        if any(r['iterations'] != saved['iterations'] for r in current):
            raise ValueError('Current hybrid iteration counts differ from the captured baseline')
        baseline_checks.append(dict(step=step, iterations=current[0]['iterations'],
            captured_iterations=saved['iterations'],
            max_common_modal_residual=max(r['common_modal_residual'] for r in current),
            median_solve_ms=statistics.median(solves[step]),
            min_solve_ms=min(solves[step]), max_solve_ms=max(solves[step])))
    native = {}
    for key in keys:
        originals = [r for r in records if r['operator'] == key and r['ordering'] == 'original']
        samples = [x for r in originals for x in r['timings'].get('native_amgx_csr', {}).get('samples_ms', [])]
        source = 'isolated native AMGX CSR (original order)'
        if not samples:
            samples = [x for p in profiles.values() for x in p.get(key, [])]
            source = 'in-hierarchy AMGX multiply CUDA events (diagnostic replay)'
        native[key] = dict(median_ms=statistics.median(samples) if samples else 0.,
                           min_ms=min(samples) if samples else 0., max_ms=max(samples) if samples else 0., source=source)
    old_work = sum(counts[k]*native[k]['median_ms'] for k in keys)
    observed_work = statistics.mean(sum(sum(v) for k, v in p.items() if k in keys) for p in profiles.values())
    consistent = old_work <= normal and observed_work <= normal
    lines = [f'Hybrid hierarchy BSR feasibility: {folder.name}', '',
        'Projection only; the integrated solver still executes its original hybrid hierarchy.', '',
        f'Normal solve mean over three RHS (five repeats each): {normal:.3f} ms; mean full timing range {normal_spread:.3f} ms.',
        f'Observed replaceable multiply event sum: {observed_work:.3f} ms ({100*observed_work/normal:.2f}% of normal solve). '
        f'Isolated-baseline model: {old_work:.3f} ms ({100*old_work/normal:.2f}%).', '',
        'Native AMGX CSR C-API products are measured for square A only. Rectangular P/R generic CSR is an explicit comparison baseline; '
        'their current solver cost comes from native in-hierarchy events. Generic CSR/BSR use the same preallocated ctypes helper. '
        'All product timings use 10 warmups and five alternating batches of 50, CUDA-event envelopes (including host submission gaps).', '',
        'Profiling serializes multiply calls. Its solve time is excluded. Fused smoother kernels, fine BSR A, '
        'vector work, reductions, coarse factorization/solve, and all unobserved work are held constant. '
        'RCM assumes vectors remain in each level’s permuted order; no per-product permutation is charged.', '',
        '| Ordering | Block | Storage × CSR | Total operator GiB | Generic CSR/BSR product speedup | Current-native/BSR product speedup | Projected solve change | Conservative saving interval (ms) | Conversion + permutation (s) | Decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    comparisons = []
    for ordering in ('original', 'rcm'):
        for block in (2, 4, 7, 8):
            group = [r for r in records if r['ordering'] == ordering and r['block_size'] == block]
            cost = sum(counts[r['operator']]*r['timings']['generic_bsr']['median_ms'] for r in group)
            generic = sum(counts[r['operator']]*r['timings']['generic_csr']['median_ms'] for r in group)
            saving = old_work-cost
            lower = sum(counts[r['operator']]*(native[r['operator']]['min_ms']-r['timings']['generic_bsr']['max_ms']) for r in group)-normal_spread
            upper = sum(counts[r['operator']]*(native[r['operator']]['max_ms']-r['timings']['generic_bsr']['min_ms']) for r in group)+normal_spread
            scalar_bytes = sum(r['storage']['csr_bytes'] for r in group)
            candidate_bytes = sum(r['storage']['bsr_bytes'] for r in group)
            inflation = candidate_bytes/scalar_bytes
            total_operator_bytes = fine_bytes+candidate_bytes
            storage_totals = {key: sum(r['storage'][key] for r in group) for key in (
                'csr_entries', 'occupied_blocks', 'occupied_block_entries', 'nonzero_entries',
                'original_explicit_zeros', 'added_interior_zeros', 'padding_entries', 'explicit_zeros',
                'padded_rows', 'padded_cols', 'vector_padding_bytes')}
            storage_totals.update(csr_bytes=scalar_bytes, bsr_bytes=candidate_bytes,
                unchanged_fine_bsr_bytes=fine_bytes, current_total_operator_bytes=fine_bytes+scalar_bytes,
                candidate_total_operator_bytes=total_operator_bytes,
                generic_bsr_workspace_bytes=sum(r['workspace_bytes']['generic_bsr'] for r in group),
                generic_csr_workspace_bytes=sum(r['workspace_bytes']['generic_csr'] for r in group))
            # Permute each operator once for an integrated candidate.
            conversion = permutation_seconds[ordering] + sum(r['conversion_seconds']+r['permutation_seconds'] for r in group)
            favorable = [r['operator'] for r in group if counts[r['operator']] and
                         r['timings']['generic_bsr']['max_ms'] < native[r['operator']]['min_ms']]
            decision = ('candidate for integration' if consistent and lower > 0 else
                        'projected slower beyond observed variability' if consistent and upper < 0 else
                        'gain not established')
            if not consistent:
                decision = 'profiling/model inconsistency; no recommendation'
            lines.append(f'| {ordering} | {block} | {inflation:.3f} | {total_operator_bytes/2**30:.3f} | {generic/cost:.3f}× | {old_work/cost:.3f}× | '
                         f'{-100*saving/normal:+.2f}% | [{lower:.3f}, {upper:.3f}] | {conversion:.3f} | {decision} |')
            comparisons.append(dict(ordering=ordering, block_size=block, storage_inflation=inflation,
                storage_totals=storage_totals,
                combined_benchmark_device_setup_seconds=sum(r['device_setup_seconds'] for r in group),
                generic_csr_speedup=generic/cost, native_speedup=old_work/cost,
                projected_solve_ms=normal-saving, projected_saving_ms=saving,
                conservative_saving_interval_ms=[lower, upper], conversion_seconds=conversion,
                locally_favorable_operators=favorable, decision=decision))
    lines += ['', 'Negative solve change means faster. Intervals use the full observed product ranges plus the normal solve timing range; '
              'they are conservative sensitivity bounds, not confidence intervals. Conversion/setup is excluded from solve projections.', '',
              'Per-operator storage, explicit zeros, padding, workspaces, three-vector checks, timing samples/spreads, and setup costs '
              'are retained in `products/*.json` and `comparison.json`. The same fine ordering is used for every candidate. '
              'Permutation construction and host conversion are recorded separately from device upload/descriptor setup.', '',
              f'Unchanged fine BSR storage: {fine_bytes/2**30:.3f} GiB. Storage inflation compares exported coarse A/P/R only. '
              'Total operator GiB also includes fine A; vectors, workspaces, setup shadows and factorization storage are separate. '
              'Device setup values in product files combine all benchmark backends and are not a BSR deployment setup estimate.', '',
              'Normal replay checks (same iteration counts as the original saved hybrid study):', '',
              '| Saved step | Iterations | Normal median ms | Normal range ms | Max modal residual |',
              '|---|---:|---:|---:|---:|']
    for check in baseline_checks:
        lines.append(f'| {check["step"]} | {check["iterations"]} | {check["median_solve_ms"]:.3f} | '
                     f'[{check["min_solve_ms"]:.3f}, {check["max_solve_ms"]:.3f}] | {check["max_common_modal_residual"]:.3e} |')
    lines += ['', 'Locally favorable operators (their full BSR timing range beats the current native baseline):']
    for c in comparisons:
        lines.append(f'- {c["ordering"]}, block {c["block_size"]}: ' + (', '.join(c['locally_favorable_operators']) or 'none') +
                     ('; entire BSR hierarchy loses in the projection.' if c['projected_saving_ms'] < 0 else '.'))
    selective = []
    lines += ['', 'Selective original-order projection (only operators with disjoint favorable native/BSR timing ranges):', '',
              '| Block | Operators | Projected saving ms | Conservative saving interval ms | Saving using generic CSR on the same operators ms |',
              '|---|---|---:|---:|---:|']
    for block in (2, 4, 7, 8):
        group = [r for r in records if r['ordering'] == 'original' and r['block_size'] == block
                 and counts[r['operator']] > 0
                 and r['timings']['generic_bsr']['max_ms'] < native[r['operator']]['min_ms']]
        if not group:
            continue
        saving = sum(counts[r['operator']]*(native[r['operator']]['median_ms']-r['timings']['generic_bsr']['median_ms']) for r in group)
        lower = sum(counts[r['operator']]*(native[r['operator']]['min_ms']-r['timings']['generic_bsr']['max_ms']) for r in group)-normal_spread
        upper = sum(counts[r['operator']]*(native[r['operator']]['max_ms']-r['timings']['generic_bsr']['min_ms']) for r in group)+normal_spread
        csr_saving = sum(counts[r['operator']]*(native[r['operator']]['median_ms']-r['timings']['generic_csr']['median_ms']) for r in group)
        operators = [r['operator'] for r in group]
        selective.append(dict(block_size=block, operators=operators, projected_saving_ms=saving,
            conservative_saving_interval_ms=[lower, upper], generic_csr_saving_ms=csr_saving,
            gain_exceeds_observed_variability=consistent and lower > 0))
        lines.append(f'| {block} | {", ".join(operators)} | {saving:.3f} | [{lower:.3f}, {upper:.3f}] | {csr_saving:.3f} |')
    lines += ['', 'Selective projections keep every unselected operator unchanged. They assume once-padded level vectors; '
              'all additional integration overhead remains unmeasured. Selection uses this experiment’s samples, so a separate integrated confirmation is required. '
              'If generic CSR is faster on the same operators, a gain over native AMGX does not establish an advantage from BSR storage itself.']
    best = min(comparisons, key=lambda c: c['projected_solve_ms'])
    if all(c['projected_saving_ms'] < 0 for c in comparisons):
        recommendation = (f'Do not implement an entirely BSR coarse hierarchy on this evidence. '
            f'The best tested combination is {best["ordering"]}, block {best["block_size"]}: '
            f'{100*(best["projected_solve_ms"]/normal-1):.2f}% projected slower, '
            f'{best["storage_inflation"]:.3f} times the coarse-A/P/R storage. '
            'Selective gains are reported separately below and do not demonstrate an integrated solver speedup.')
    else:
        recommendation = 'Use the variability gate and selective results below to decide which candidates warrant integrated confirmation.'
    lines[2:2] = [recommendation, '']
    save(folder/'projection.json', dict(comparisons=comparisons, selective_original=selective,
        recommendation=recommendation, normal_solve_ms=normal,
        normal_solve_spread_ms=normal_spread, measured_multiply_ms=observed_work, baseline_checks=baseline_checks,
        modeled_multiply_ms=old_work, counts_per_solve=counts, current_native_baselines=native,
        profile_model_consistent=consistent, integrated_solver_accelerated=False))
    (folder/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'baseline', 'measure'), default='all')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for triangles, systems in SYSTEMS.items():
        meta = json.loads((systems/'metadata.json').read_text())
        if meta['triangles'] != triangles or meta['config']['order'] != 6:
            raise ValueError('Captured mesh or polynomial degree does not match study')
        if len(list(systems.glob('system_step*.npz'))) != 3:
            raise ValueError('Require exactly three saved systems')
        folder = args.output_dir/f'p6_{triangles}'
        folder.mkdir(exist_ok=True)
        if args.phase in ('all', 'baseline'):
            replay(systems, folder/'baseline')
        if args.phase in ('all', 'measure'):
            if not (folder/'baseline/completion.json').exists():
                raise ValueError('Run --phase baseline first')
            replay(systems, folder/'diagnostic', diagnostic=True)
            measure(folder)
    if args.phase != 'baseline':
        (args.output_dir/'report.md').write_text('\n\n'.join(
            (args.output_dir/f'p6_{triangles}/report.md').read_text() for triangles in SYSTEMS))
    print(f'Completed {args.phase}: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
