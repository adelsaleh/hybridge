"""Expanded H200 experiment design and complete registered solver catalogue.

Planning only: no numerical imports, submission, or fabricated runtime estimates.
All registered configurations are screened per HDG degree before scaling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

from campaigns.adr_unified.adr_unified_plan import configure_amgx_backend, fingerprint, problem_class, read_inventory
from campaigns.common.production_placement import resources

ROOT = Path(__file__).resolve().parents[2]
DEGREES = tuple(range(1, 10))


def _load_common():
    spec = importlib.util.spec_from_file_location('_expanded_adr_common', ROOT / 'vendor/adr_gmres/scripts/adr_performance_common.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def solver_catalog(inventory):
    catalog = {}

    def add(solver, source, equations, *, matrix_format='bsr', host_precompute=False, note=None):
        solver = deepcopy(solver)
        if 'amgx_config' in solver:
            solver['amgx_config'] = configure_amgx_backend(solver['amgx_config'], 'cusparse_generic')
        identity = fingerprint(dict(solver=solver, matrix_format=matrix_format))
        row = catalog.setdefault(identity, dict(id=identity[:20], solver_sha256=identity, solver=solver,
            sources=[], equations=[], degrees=list(DEGREES), matrix_format=matrix_format,
            requires_host_precompute=host_precompute, compatibility='per-degree probe required', notes=[]))
        row['sources'] = sorted(set(row['sources']) | {source})
        row['equations'] = sorted(set(row['equations']) | set(equations))
        row['requires_host_precompute'] |= host_precompute or solver['family'] == 'native_hp'
        if note and note not in row['notes']:
            row['notes'].append(note)

    def amgx_equations(config):
        outer = config.get('solver', {})
        outer = outer.get('solver') if isinstance(outer, dict) else outer
        return ('poisson',) if outer in {'PCG', 'PCGF', 'CG', 'FCG'} else ('poisson', 'transport', 'adr')

    for system in inventory['systems']:
        for choice in system['candidates']:
            solver = choice['solver']
            family = solver['family']
            equations = ('poisson', 'adr') if family == 'native_hp' else ('poisson', 'transport', 'adr')
            if family == 'amgx':
                equations = amgx_equations(solver['amgx_config'])
            add(solver, f"inventory:{choice['id']}", equations,
                matrix_format=solver.get('matrix_format', 'bsr'),
                host_precompute=solver.get('configuration', {}).get('local_solver') == 'cpu_inverse',
                note='Archived basis/layout must be checked by the degree probe; no implicit basis conversion.')
    for path in sorted((ROOT / 'configs/amgx').glob('*.json')):
        config = json.loads(path.read_text())
        outer = config.get('solver', {})
        note = None
        if isinstance(outer, dict) and outer.get('solver') == 'BICGSTAB' and outer.get('preconditioner'):
            note = 'AMGX BICGSTAB ignores nested preconditioner in this stack; report effective preconditioning, not filename.'
        add(dict(family='amgx', amgx_config=config), str(path.relative_to(ROOT)), amgx_equations(config),
            matrix_format='bsr' if 'bsr' in path.stem else 'csr', note=note)
    common = _load_common()
    # All implementation paths supported by the existing builder, without an
    # unbounded Cartesian product of every numerical tuning parameter.
    for config in common.path_configurations():
        family = {'none': 'unpreconditioned', 'poly': 'pp', 'asm': 'asm', 'asm_poly': 'asm_pp',
                  'block_jacobi': 'bj', 'block_jacobi_poly': 'bj_pp'}[config.preconditioner]
        add(dict(family=family, configuration=asdict(config), maxiter=2000),
            f"path_configurations:{config.key}", ('poisson', 'transport', 'adr'),
            host_precompute=config.local_solver == 'cpu_inverse',
            note='CPU inverse construction, when selected, is a separate CPU-node dependency; GPU setup only uploads it.')
    for policy in ('standard', 'fast', 'robust'):
        add(dict(family='native_hp', policy=policy, maxiter=2000, restart=75), f'native-pmg:{policy}',
            ('poisson', 'adr'), note='Fast policy outside its measured p6 regime is an experimental degree probe.')
    for method in ('cg', 'bicgstab', 'cgs', 'gmres'):
        add(dict(family='cupyx', method=method, maxiter=2000), f'cupyx:{method}',
            ('poisson',) if method == 'cg' else ('poisson', 'transport', 'adr'), matrix_format='csr')
    return sorted(catalog.values(), key=lambda row: row['id'])


def problem_cases(inventory):
    cases = {}
    for system in inventory['systems']:
        key = problem_class(system)
        if key not in cases or system['problem']['p'] > cases[key]['template']['p']:
            cases[key] = dict(id='adr-' + fingerprint(key)[:16], equation='adr', geometry=system['geometry'],
                regime=system['problem']['case'], template=system['problem'],
                source_system_id=system['id'], source_mesh=system['mesh'], degrees=list(DEGREES))
    result = list(cases.values())
    for geometry in ('square', 'disk', 'annulus', 'five-lobed-hole'):
        result.append(dict(id=f'poisson-{geometry}', equation='poisson', geometry=geometry, regime='pure-poisson',
            source='constant and manufactured RHS', boundary='zero Dirichlet', degrees=list(DEGREES),
            stabilization_controls=[1., 128000.]))
    for case in ('test2', 'disk-tangent'):
        result.append(dict(id=f'transport-{case}', equation='transport', geometry='square' if case == 'test2' else 'disk',
            regime=case, factory='scripts.gpu.benchmark_advection_tsle_bsr._problem', degrees=list(DEGREES),
            boundary='Dirichlet' if case == 'test2' else 'zero-flux', reaction_note='transport operator including the time-step mass'))
    for kind in ('scalar', 'diagonal', 'symmetric', 'general'):
        result.append(dict(id=f'adr-tensor-{kind}', equation='adr', geometry='square', regime='manufactured-tensor',
            factory='scripts.advection_diffusion_reaction.cases.tensor_cases.manufactured_tensor',
            factory_arguments=dict(kind=kind), degrees=list(DEGREES)))
    return sorted(result, key=lambda row: row['id'])


def design(inventory_path):
    inventory = read_inventory(inventory_path)
    catalog, cases = solver_catalog(inventory), problem_cases(inventory)
    eligible_per_equation = {equation: sum(equation in row['equations'] for row in catalog)
                             for equation in ('poisson', 'transport', 'adr')}
    screening = sum(eligible_per_equation[case['equation']] * len(DEGREES) for case in cases)
    return dict(schema_version=1, launch_ready=False, scope='expanded H200 campaign authorized 2026-10-26',
        previous_dossier_is_budget_cap=False,
        measured_costs_available=False, degrees=list(DEGREES), solver_catalog=catalog, cases=cases,
        counts=dict(configurations=len(catalog), by_family=dict(Counter(row['solver']['family'] for row in catalog)),
            cases_by_equation=dict(Counter(case['equation'] for case in cases)),
            configurations_by_equation=eligible_per_equation, screening_trials=screening,
            four_size_scaling_trials=4*screening,
            note='Candidate/degree eligibility counts before feasibility and memory probes; not submitted jobs or runtime estimates.'),
        stages=[
            dict(name='screening', purpose='every eligible registered config at every degree and problem class',
                 target_triangles=2048, repeats=3, warmups=1),
            dict(name='scaling', purpose='four meshes per degree using measured path-specific memory',
                 usable_memory_fractions=[.03, .15, .40, .75], repeats=7, warmups=2),
            dict(name='capacity', purpose='bracket per-path capacity and test solver behavior near the limit',
                 search='geometric growth then bounded bisection', max_device_memory_fraction=.90,
                 reserve_device_gib=8, max_probes_per_path=14,
                 stop_on=['memory budget', 'shared memory limit', 'per-trial timeout'],
                 do_not_prune_for=['nonconvergence', 'slower than PARDISO', 'previous local failure']),
            dict(name='tuning', purpose='bounded parameter sweeps at each degree, then independent confirmation',
                 polynomial_degrees=[8, 16, 24, 48, 96], restarts=[30, 75, 150],
                 orthogonalizations=['mgs', 'mgs2', 'cgs', 'cgs2'], pmg_policies=['standard', 'fast', 'robust'],
                 iteration_caps=[2000, 10000], cpu_pardiso_threads=[1, 8, 16]),
            dict(name='guiding-center-replay', purpose='confirm choices with captured Poisson and changing transport operators',
                 inputs='saved early/filamented/saturated states', time_schemes=['si-bdf2', 'si-bdf3', 'imex-ark3'],
                 required_before_execution='captured immutable systems and physical RHS; no synthetic substitute for conclusions')],
        measurements=['assembly', 'format conversion', 'upload', 'preconditioner setup', 'cold zero-guess solve',
            'reused setup with zero guess', 'reused setup with previous-solution guess', 'reconstruction',
            'flux postprocessing', 'primal postprocessing', 'true physical residual', 'iterations',
            'GPU peak memory', 'host peak RSS', 'factor fill', 'observed CPU threads',
            'kernel block size', 'registers', 'shared memory', 'local memory/spills', 'measured Nsight counters'],
        fairness=dict(precision='FP64; FP32 kernel comparisons are diagnostics only',
            physical_relative_residual=1e-10, tolerance_control=1e-12,
            stopping_rule='normalize configured stopping thresholds to the common physical target; freeze and hash the effective JSON before execution',
            scaling='CG/PCG requires unscaled or symmetric scaling; left scaling is a separate nonsymmetric control',
            matrix_identity='one immutable matrix/RHS per case, degree, mesh and basis; canonical and topology hashes',
            layout_control='CSR/BSR are storage views of the same system; basis changes require distinct references',
            direct_reference='one CPU PARDISO reference per distinct system, not per iterative configuration',
            direct_capacity_failures=['host memory', 'sparse index width', 'factorization failure', 'timeout'],
            beyond_direct_capacity='retain validated GPU results; no speedup ratio without a completed CPU reference',
            no_fallback=True, retain_failures=True, setup_and_reuse_reported_separately=True,
            statistical_reporting='raw samples, median and IQR; independent repeats for close rankings'),
        placement=dict(host=resources('host'), device=resources('device'), host_on_gpu_nodes_allowed=False,
            host_inverse_rule='CPU inverse variants require a separately prepared CPU artifact and must not execute CPU inversions on GPU nodes'),
        unsupported=[dict(equation='transport', family='native_hp', reason='No validated native pMG transport path; record unsupported instead of fallback.')],
        reports=dict(winners_group_by=['equation', 'regime', 'geometry', 'degree', 'mesh', 'reuse_scenario'],
            amortization_solves=[1, 10, 100, 1000],
            guiding_center_objective='sum of assembly + setup + solve + recovery at the actual scheme stage count',
            conclusions='No universal fastest configuration without workload/degree/reuse qualifiers.'),
        required_before_launch=['passing smoke', 'degree/format capability probes', 'memory-calibrated system generation',
            'separate host/device runners and cached-system adapters', 'storage/quota preflight per batch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=ROOT / 'run_configs/adr_unified_l5/inventory.json')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    report = design(args.inventory)
    if args.summary:
        report.pop('solver_catalog')
        report.pop('cases')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
