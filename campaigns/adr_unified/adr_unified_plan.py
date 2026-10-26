"""Pure, portable planning for all archived ADR systems plus a large L5.

No numerical imports: usable on login nodes without a CUDA context. Candidate
IDs hash complete solver definitions, not labels or historical pass/fail status.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

AMGX_BACKENDS = ('cusparse_generic', 'legacy')


def configure_amgx_backend(config, backend):
    """Copy a campaign config, changing only explicit BSR backend selectors.

    AMGX scopes this option: an outer override alone does not override a nested
    preconditioner/smoother. Preserve every numerical setting and make the root
    choice explicit, including for archived configs without a root selector.
    """
    if backend not in AMGX_BACKENDS:
        raise ValueError(f'Unsupported AMGX BSR backend: {backend}')

    def rewrite(value):
        if isinstance(value, dict):
            return {k: backend if k == 'bsr_spmv_backend' else rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        return deepcopy(value)

    result = rewrite(config)
    root_scope = result['solver'] if isinstance(result.get('solver'), dict) else result
    root_scope['bsr_spmv_backend'] = backend
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_inventory(path):
    path = Path(path).resolve()
    inventory = json.loads(path.read_text())
    if inventory.get('schema_version') != 1 or not inventory.get('systems'):
        raise ValueError('Unsupported or empty ADR inventory')
    seen = set()
    checked_meshes = set()
    for system in inventory['systems']:
        if system['id'] in seen:
            raise ValueError('Duplicate system identity')
        seen.add(system['id'])
        if not system['candidates']:
            raise ValueError('System without solver configurations')
        for choice in system['candidates']:
            if 'pardiso' in json.dumps(choice).lower():
                raise ValueError('Direct solver found in iterative inventory')
        mesh = system.get('mesh')
        if mesh and mesh['file'] not in checked_meshes:
            asset = (path.parent/mesh['file']).resolve()
            if not asset.is_relative_to(path.parent):
                raise ValueError('Mesh path escapes inventory directory')
            with asset.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != mesh['sha256']:
                raise ValueError(f'Changed mesh asset {asset}')
            checked_meshes.add(mesh['file'])
    return inventory


def problem_class(system):
    p = system['problem']
    return (system['geometry'], p['case'], fingerprint(p.get('stress_parameters', {})))


def build_plan(inventory, *, l5_triangles=400000, maxiter=2000, repeats=3, warmup=1,
               amgx_backend='cusparse_generic'):
    if amgx_backend not in AMGX_BACKENDS:
        raise ValueError(f'Unsupported AMGX BSR backend: {amgx_backend}')
    if l5_triangles < 300000:
        raise ValueError('L5 must be at least 300,000 triangles; do not relabel L4')
    if maxiter < 2000 or repeats < 1 or warmup < 0:
        raise ValueError('Require maxiter >= 2000, repeats >= 1, warmup >= 0')
    systems = deepcopy(inventory['systems'])
    groups = defaultdict(list)
    for system in systems:
        system['level'] = 'original'
        p = system['problem']
        if p['p'] == 6 and system['geometry'] == 'square' and p['n'] in (32, 64, 128, 223):
            system['level'] = 'L'+str((32, 64, 128, 223).index(p['n'])+1)
        elif p['p'] == 6 and system['geometry'] == 'annulus' and system['triangles'] in (2043, 8039, 33174, 99984):
            system['level'] = 'L'+str((2043, 8039, 33174, 99984).index(system['triangles'])+1)
        groups[problem_class(system)].append(system)
    for group in groups.values():
        # Keep the most resolved p=6 quadrature definition for each PDE class.
        base = max(group, key=lambda s: (s['problem']['p'], s['triangles']))
        system = deepcopy(base)
        system.pop('archived_operator_sha256', None)
        system.update(level='L5', target_triangles=l5_triangles, trace_dofs=None)
        system['problem']['p'] = 6
        # Explicit quadrature from lower-order exploratory controls must not
        # underintegrate their new p=6 extension.
        for key in ('volume_quad_1d', 'edge_quad_1d'):
            value = system['problem'].get(key)
            if value is not None:
                system['problem'][key] = max(value, 14)
        n = round(math.sqrt(l5_triangles/2))
        system['problem']['n'] = n
        if system['geometry'] == 'square':
            system['mesh'] = None
            system['triangles'] = 2*n*n
            system['trace_dofs'] = (3*n*n-2*n)*7
        else:
            stress = 'stress_parameters' in system['problem']
            system['mesh'] = dict(generator='stress' if stress else 'five_lobed',
                                  target_triangles=l5_triangles,
                                  max_triangles=math.ceil(1.03*l5_triangles),
                                  boundary_points=3600 if stress else 1600,
                                  neck_elements=12 if stress else None)
            system['triangles'] = None
        candidates = {}
        for source in group:
            for choice in source['candidates']:
                candidates.setdefault(choice['id'], deepcopy(choice))
        system['candidates'] = sorted(candidates.values(), key=lambda c: c['id'])
        system['origins'] = [dict(suite='L5', source_systems=[s['id'] for s in group])]
        system['id'] = 'L5_'+fingerprint(dict(problem=system['problem'], geometry=system['geometry'], mesh=system['mesh']))[:16]
        systems.append(system)
    for system in systems:
        for choice in system['candidates']:
            # Explicit shared iteration cap; retain archived restart, PP and
            # hierarchy settings instead of retuning to suit device memory.
            choice['solver']['maxiter'] = maxiter
            if 'amgx_config' in choice['solver']:
                choice['solver']['amgx_config'] = configure_amgx_backend(
                    choice['solver']['amgx_config'], amgx_backend)
            # Keep archival candidate IDs/aliases for cross-campaign joins;
            # separately identify the effective, backend-specific definition.
            choice['effective_solver_sha256'] = fingerprint(choice['solver'])
    protocol = dict(rtol=1e-10, internal_rtol=1e-11, maxiter=maxiter,
                    warmup=warmup, repeats=repeats, solves_per_setup=2,
                    initial_guess='zero for every solve', precision='float64')
    coverage = dict(original_systems=len(inventory['systems']), l5_systems=len(groups),
                    total_systems=len(systems), solver_jobs=sum(len(s['candidates']) for s in systems),
                    jobs_by_family=dict(Counter(c['solver']['family'] for s in systems for c in s['candidates'])),
                    original_suites=sorted({o['suite'] for s in inventory['systems'] for o in s['origins']}))
    return dict(schema_version=1, inventory_sha256=fingerprint(inventory), protocol=protocol,
                amgx_backend=amgx_backend,
                l5_triangles=l5_triangles, systems=systems, coverage=coverage)


def memory_estimate(system, choice):
    """Conservative planning bytes, not a guarantee or measured high-water mark.

    Include mixed local caches for CPU preparation and an explicit 2-vector
    Arnoldi basis for flexible GMRES. AMG fill-in is data dependent, so reserve
    an additional hierarchy allowance and retain runtime OOM as a distinct
    result. Re-evaluate after the real mesh and free device memory are known.
    """
    triangles = system.get('triangles') or math.ceil(1.03*system['target_triangles'])
    p = system['problem']['p']
    f, local = p+1, (p+1)*(p+2)//2
    faces = math.ceil(1.5*triangles+4*math.sqrt(triangles))
    dofs = system.get('trace_dofs') or faces*f
    solver = choice['solver']
    config = solver.get('configuration', {})
    restart = max(solver.get('restart', 75), config.get('restart', 75), config.get('polynomial_degree') or 0)
    blocks = faces*5*f*f*8
    patches = triangles*9*f*f*8
    basis = dofs*(2*restart+16)*8
    mixed = triangles*(3*local)**2*8
    coupling = triangles*9*local*f*8
    hierarchy = 8*blocks if solver['family'] in ('amgx', 'native_hp') else 0
    return dict(solver_device_bytes=3*blocks+3*patches+basis+hierarchy+512*2**20,
                assembly_device_bytes=4*mixed+5*coupling+4*patches+3*blocks,
                assembly_host_bytes=6*mixed+6*coupling+4*patches+6*blocks+32*triangles*(2*p+2)**2*8,
                solver_host_bytes=mixed+coupling+4*patches+6*blocks,
                cache_disk_bytes=mixed+coupling+2*patches+blocks+triangles*3*local*8,
                planning_only=True)
