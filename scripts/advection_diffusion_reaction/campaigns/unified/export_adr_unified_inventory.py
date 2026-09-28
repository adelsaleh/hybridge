#!/usr/bin/env python3
"""Freeze archived ADR specifications into a portable rerun inventory.

Reads specifications and copies only mesh assets, never assembled matrices or
timing samples. This maintenance/export step is not needed on the cluster.
Neither importing nor running it imports numerical packages or launches solves.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
STUDY = ROOT / 'run_outputs/solver_studies/adr_scaling_2026_09_17'
COMMON_KEYS = ('case', 'n', 'p', 'volume_quad_1d', 'edge_quad_1d',
               'stress_parameters', 'velocity_normalization', 'normalization_record')
SOLVER_KEYS = ('family', 'configuration', 'amgx_config', 'matrix_format', 'policy',
               'native_tuning', 'native_configuration', 'restart')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def is_solver(spec):
    return (bool(spec.get('candidate')) and bool(spec.get('family'))
            and spec.get('task') in (None, 'measure') and not spec.get('profile')
            and spec.get('worker_kind') not in ('profile', 'native_profile', 'amgx_profile')
            and 'pardiso' not in spec['candidate'].lower())


def export(output, native_manifest, stress_root):
    manifest = read(native_manifest)
    systems = deepcopy(manifest['systems'])
    roots = {Path(o['campaign_root']) for s in systems for o in s['origins']}
    roots.add(Path(native_manifest).parent)
    roots.add(stress_root)
    paths = set(p for root in roots for p in (root/'specs').glob('*.json'))
    paths.update((STUDY/'oscillatory/asm_optimality_audit').glob('*.spec.json'))
    paths.update(Path(p) for p in manifest.get('source_artifacts', {}) if p.endswith('.spec.json'))
    indexed = defaultdict(list)
    for path in sorted(paths):
        spec = read(path)
        if is_solver(spec):
            indexed[str(Path(spec['cache']).resolve())].append((path, spec))
    for path in sorted((stress_root/'specs').glob('assemble_*.json')):
        spec = read(path)
        cache = str(Path(spec['cache']).resolve())
        rows = indexed[cache]
        if not rows:
            raise ValueError(f'No stress solver specifications for {path}')
        summary = read(stress_root/'summary.json')
        result = next(r for r in summary if r.get('operator_sha256') == rows[0][1]['expected_operator_sha256'])
        systems.append(dict(system_id=rows[0][1]['expected_operator_sha256'],
                            cache=cache, case=spec['case'], p=spec['p'],
                            geometry=spec['stress_parameters'].get('geometry', 'annulus'),
                            triangles=result['triangles'], trace_dofs=result['trace_dofs'],
                            origins=[dict(suite='stress', campaign=stress_root.name)]))
    output.mkdir(parents=True, exist_ok=True)
    (output/'meshes').mkdir(exist_ok=True)
    entries, sources = [], {}
    for system in systems:
        candidates = {}
        rows = indexed[str(Path(system['cache']).resolve())]
        if not rows:
            raise ValueError(f'No specifications for {system["system_id"]}')
        common = None
        mesh = None
        for path, spec in rows:
            current = {k: spec[k] for k in COMMON_KEYS if k in spec}
            # Omitted quadrature means the same worker default as explicit null.
            current.setdefault('volume_quad_1d', None)
            current.setdefault('edge_quad_1d', None)
            if common is not None and common != current:
                raise ValueError(f'Conflicting problem definitions at {path}')
            common = current
            mesh_path = spec.get('mesh_path')
            current_mesh = None
            if mesh_path:
                source = Path(mesh_path)
                sha = file_hash(source)
                destination = output/'meshes'/f'{sha}.npz'
                if not destination.exists():
                    shutil.copy2(source, destination)
                elif file_hash(destination) != sha:
                    raise ValueError(f'Corrupt exported mesh {destination}')
                current_mesh = dict(file=f'meshes/{sha}.npz', sha256=sha)
            if mesh is not None and mesh != current_mesh:
                raise ValueError(f'Conflicting meshes at {path}')
            mesh = current_mesh
            choice = {k: spec[k] for k in SOLVER_KEYS if k in spec}
            choice.setdefault('restart', spec.get('configuration', {}).get('restart', 75))
            key = digest(choice)[:16]
            candidate = candidates.setdefault(key, dict(id=key, solver=choice, aliases=[]))
            if spec['candidate'] not in candidate['aliases']:
                candidate['aliases'].append(spec['candidate'])
            sources[f'{path.parent.parent.name}/{path.name}'] = file_hash(path)
        # Both published policies must be present even if the earlier completion
        # reused a result and consequently did not emit a new job specification.
        if 'stress_parameters' not in common:
            for policy in ('standard', 'robust'):
                choice = dict(family='native_hp', policy=policy, restart=75)
                key = digest(choice)[:16]
                candidates.setdefault(key, dict(id=key, solver=choice, aliases=['native_hp_'+policy]))
        entries.append(dict(id=system['system_id'][:16], archived_operator_sha256=system['system_id'],
                            problem=common, geometry=system['geometry'], mesh=mesh,
                            triangles=system['triangles'], trace_dofs=system['trace_dofs'],
                            origins=[{k: o[k] for k in ('suite', 'campaign', 'nominal_n') if k in o}
                                     for o in system['origins']],
                            candidates=sorted(candidates.values(), key=lambda c: c['id'])))
    payload = dict(schema_version=1, systems=sorted(entries, key=lambda e: e['id']),
                   source_specification_sha256=sources,
                   policy='All archived iterative specifications, including failures; no timing-based filtering. '
                          'Original meshes retained; no old matrix caches or absolute paths required.')
    serialized = json.dumps(payload, indent=2, sort_keys=True)+'\n'
    if '/home/' in serialized:
        raise ValueError('Machine-specific path escaped into portable inventory')
    target = output/'inventory.json'
    if target.exists() and target.read_text() != serialized:
        raise ValueError('Refusing to overwrite a different frozen inventory; use a new output')
    target.write_text(serialized)
    print(json.dumps(dict(systems=len(entries), solver_jobs=sum(len(e['candidates']) for e in entries),
                          mesh_assets=len(list((output/'meshes').glob('*.npz'))), inventory=str(target)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-manifest', type=Path,
                        default=STUDY/'native_hp_completion_2026_09_21/manifest.json')
    parser.add_argument('--stress-root', type=Path,
                        default=ROOT/'run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k')
    args = parser.parse_args()
    export(args.output.resolve(), args.native_manifest, args.stress_root)


if __name__ == '__main__':
    main()
