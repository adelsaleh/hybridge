#!/usr/bin/env python3
"""Audit saved oscillatory ADR measurements, without rerunning any solver."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scripts.adr_performance_common import atomic_json, read_json

CAMPAIGNS = (
    'adr_oscillatory_screen_20260918',
    'adr_oscillatory_square_p6_20260918',
    'adr_oscillatory_star_h0.04_p6_20260918',
    'adr_oscillatory_star_h0.02_p6_20260918',
)


def sha256(path):
    """Hash large binary artifacts without loading them fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path):
    """Record the checkout identifier; source snapshots also cover dirty files."""
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    """Verify complete samples, identical inputs, saved solutions and diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    branch = args.branch.resolve()
    master = branch.parent / 'hdgfem'
    verify = branch / 'run_logs/adr_oscillatory_verification_20260918'
    checks, all_rows, assemblies, controls, source_checks = [], [], [], [], []
    for name in CAMPAIGNS:
        folder = branch / 'run_logs' / name
        manifest = read_json(folder / 'manifest.json')
        completed = read_json(folder / 'completion.json')
        rows = read_json(folder / 'summary.json')
        assert completed['status'] == 'completed'
        assert completed['attempted'] == completed['scheduled'] == len(rows)
        assert completed['passed'] == sum(r['status'] == 'passed' for r in rows)
        assert len(rows) == len(manifest['arguments']['cases'])*len(manifest['candidates'])
        all_rows.extend(rows)
        assembled = read_json(folder / 'assemblies.json')
        assert len(assembled) == len(manifest['arguments']['cases'])
        assert all(a['status'] == 'passed' for a in assembled)
        assemblies.extend(assembled)
        for script, digest in manifest['script_sha256'].items():
            assert sha256(folder / 'source_snapshot' / script) == digest
            current = sha256(branch / 'scripts' / script)
            if current != digest:
                assert name == CAMPAIGNS[0] and script in ('run_oscillatory_adr_study.py', 'adr_performance_worker.py')
            source_checks.append(dict(campaign=name, script=script, archived_sha256=digest,
                                      current_sha256=current, matches_current=current == digest))
        aggregate = hashlib.sha256()
        files = list((branch/'hdgfem').rglob('*.py')) + [branch/'scripts'/script for script in
            ('run_adv_diff_rea_performance.py', 'adr_performance_worker.py',
             'adr_performance_common.py', 'adv_diff_rea_cases.py')]
        for path in sorted(set(files)):
            aggregate.update(str(path.relative_to(branch)).encode())
            archived = folder/'source_snapshot'/path.name
            aggregate.update((archived if path.parent == branch/'scripts' and archived.exists() else path).read_bytes())
        assert aggregate.hexdigest() == manifest['source_sha256'], 'Changed numerical package source'
        diagnostics = folder / 'matrix_diagnostics.json'
        if diagnostics.exists():
            for row in read_json(diagnostics):
                if 'same_matrix_as_smooth_reference' in row:
                    assert row['same_matrix_as_smooth_reference']
                    controls.append(dict(campaign=name, **row))
        for case in manifest['arguments']['cases']:
            group = [r for r in rows if r['case'] == case]
            assert len(group) == len(manifest['candidates'])
            assert len({r['candidate'] for r in group}) == len(group)
            cache = Path(group[0]['matrix_cache'])
            blocks = np.load(cache / 'system_blocks.npy', mmap_mode='r')
            rhs = np.load(cache / 'system_rhs.npy', mmap_mode='r')
            digest = hashlib.sha256(blocks.tobytes() + rhs.reshape(-1).tobytes()).hexdigest()
            assert {r['operator_sha256'] for r in group} == {digest}
            reference = None
            for r in group:
                key = f"{case}_n{r['n']}_p{r['p']}_{r['candidate']}"
                spec = read_json(folder / 'specs' / (key + '.json'))
                assert Path(spec['cache']) == cache == Path(r['matrix_cache'])
                assert r['worker_sha256'] == manifest['script_sha256']['adr_solver_comparison_worker.py']
                assert spec['rtol'] == r['rtol'] == 1e-10
                assert spec['internal_rtol'] == r['internal_rtol'] == 1e-11
                assert spec['maxiter'] == 1000 and spec['solves_per_setup'] == 2
                if r['family'] == 'amgx':
                    assert spec['matrix_format'] == r['matrix_format'] == 'bsr'
                if r['status'] != 'passed':
                    assert r['status'] == 'numerical_failure'
                    assert any(not s['passed'] for t in r['warmups'] + r['samples'] for s in t['solves'])
                    continue
                assert len(r['samples']) == manifest['arguments']['repeats']
                assert len(r['warmups']) == manifest['arguments']['warmup'] == 1
                for trial in r['samples'] + r['warmups']:
                    assert trial['passed'] and len(trial['solves']) == 2
                    assert all(s['passed'] and np.isfinite(s['true_relative_residual'])
                               and s['true_relative_residual'] <= 1e-10 for s in trial['solves'])
                if r['candidate'].startswith('asm_d'):
                    reference = np.load(folder / 'jobs' / (key + '.solution.npy'))
            assert reference is not None
            deviations = {}
            for r in group:
                if r['status'] != 'passed':
                    continue
                key = f"{case}_n{r['n']}_p{r['p']}_{r['candidate']}"
                solution = np.load(folder / 'jobs' / (key + '.solution.npy'))
                assert solution.size == rhs.size and np.isfinite(solution).all()
                deviations[r['candidate']] = float(np.linalg.norm(solution-reference)/np.linalg.norm(reference))
            checks.append(dict(campaign=name, case=case, trace_dofs=int(rhs.size),
                operator_rhs_sha256=digest, matrix_input_paths_match=True,
                trace_relative_difference_to_asm=deviations))
    solves = [s for r in all_rows if r['status'] == 'passed'
              for t in r['warmups']+r['samples'] for s in t['solves']]
    attempted = [s for r in all_rows for t in r['warmups']+r['samples'] for s in t['solves']]
    quadrature = [read_json(verify / ('quadrature_'+case) / 'comparison.json')
                  for case in ('cellular7_anisotropic', 'cellular7_weak')]
    assert all(q['status'] == 'passed' for q in quadrature)
    probes = read_json(verify / 'quadrature_solvers/summary.json')
    coarse = read_json(branch / 'run_logs' / CAMPAIGNS[2] / 'summary.json')
    sensitivity = []
    for probe in probes:
        baseline = next(r for r in coarse if r['case'] == 'cellular7_anisotropic' and r['candidate'] == probe['candidate'])
        iterations = lambda r: sorted({s['iterations'] for t in r['warmups']+r['samples'] for s in t['solves']})
        assert probe['status'] == baseline['status']
        assert iterations(probe) == iterations(baseline)
        sensitivity.append(dict(candidate=probe['candidate'], status=probe['status'],
            original_iterations=iterations(baseline), doubled_quadrature_iterations=iterations(probe)))
    profiles = {h: read_json(verify / ('asm_profile_h'+h+'.json')) for h in ('0.04', '0.02')}
    assert all(p['status'] == 'passed' and p['validation']['passed'] for p in profiles.values())
    geometries = []
    for path in sorted((branch / 'run_logs/adr_oscillatory_geometry_20260918').glob('star_*.json')):
        meta = read_json(path)
        assert sha256(path.with_suffix('.npz')) == meta['mesh_sha256']
        assert meta['boundary_components'] == 2 and meta['connected_components'] == 1
        geometries.append(meta)
    previous = read_json(master / 'docs/research/solver_studies/adr_solver_comparison_2026_09_17.data.json')['provenance']
    binaries = {}
    for name, expected in previous['binaries'].items():
        current = sha256(name)
        assert current == expected['sha256'], 'Changed prebuilt runtime: '+name
        binaries[name] = dict(sha256=current, size_bytes=Path(name).stat().st_size,
                             matches_prior_campaign=True)
    references = [a for a in assemblies if 'reference_l2' in a['validation']]
    result = dict(status='passed', configurations=len(all_rows),
        passed_configurations=sum(r['status']=='passed' for r in all_rows),
        failed_configurations=sum(r['status']!='passed' for r in all_rows),
        systems=len(assemblies), passing_configuration_solves=len(solves), attempted_solves=len(attempted),
        worst_physical_relative_residual=max(s['true_relative_residual'] for s in solves),
        worst_cross_solver_trace_relative_difference=max(d for c in checks for d in c['trace_relative_difference_to_asm'].values()),
        independent_cpu_reference_systems=len(references),
        worst_cpu_gpu_blocks_relative_error=max(a['validation']['cpu_gpu_blocks_relative_error'] for a in references),
        worst_cpu_gpu_rhs_relative_error=max(a['validation']['cpu_gpu_rhs_relative_error'] for a in references),
        matrix_and_worker_checks=checks, unchanged_matrix_controls=controls,
        source_snapshots=source_checks, aggregate_numerical_source_hashes_verified=True, source_change_note='After the square screening only, the runner and get_space gained optional NPZ mesh input; the rectangular fallback and numerical solver worker remained unchanged. Archived and current differences were inspected.',
        quadrature=quadrature, doubled_quadrature_solver_checks=sensitivity,
        asm_profiles=profiles, geometry=geometries,
        binaries=binaries, prior_binding_ldd=previous['binding_ldd'],
        repositories={str(p):git_head(p) for p in (branch, master)},
        cpu_tests=dict(command="python -m pytest -q tests/test_oscillatory_adr_cases.py tests/test_adv_diff_rea.py -k 'not gpu'",
                       passed=21, deselected=9, note='Executed separately after the campaign; no native build.'),
        note='Condition numbers are reproducible 1-norm lower estimates in Bernstein coordinates. Fine systems use physical residual and analytic PDE error; CPU assembly/direct references are limited to <=100,000 trace DOFs. Native libraries are prebuilt, locally modified AMGX/pyamgx artifacts; hashes identify actual binaries, not pristine upstream releases.')
    atomic_json(verify / 'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,(str,int,float)) and k not in ('prior_binding_ldd','source_change_note','note')}, indent=2))


if __name__ == '__main__':
    main()
