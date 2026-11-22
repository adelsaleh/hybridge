"""Pure planning for the H200 tuned ADR campaign (`adr_h200_tuned`): iterative vs LU ladder.

The ladder reuses the unified campaign's system definitions, so every system it shares with
the unified device campaign (21958) and its PyPardiso reference (21959) has the same plan id
(``key`` in the PARDISO records): the archived L1-L4 meshes, the 400k-triangle L5 mesh, and
the trace-DOF-sized L6 (8M) and L7 (20M) levels. Two levels beyond any LU reach, L8 (40M) and
L9 (80M), probe the H200's own limit; a system that does not fit is a measured
``memory_excluded`` or ``out_of_memory`` outcome. The opt-in level ``Lmax`` gives each degree the
largest trace-DOF target whose system job still fits one H200, the same for every class.

Candidates are the fastest converging ones per problem class from the unified campaign
(1636 jobs by 2026-10-28), with the H200 solver-kernel options of branch h200-solver-kernels.
Candidates that never or rarely converged there are left out (see ``EXCLUDED``). Every system
also gets one ``raw_assembly`` row: the tuned raw-CUDA ADR tensor assembly (branch
h200-asm-tuning, per-degree sm_90 settings) timed on the same mesh, degree and case. Its
discretization differs from the vendored one (trace basis, quadrature, stabilization), so it
is an assembly cost reference and never feeds the solvers.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from copy import deepcopy

from campaigns.adr_h200_tuned.device_system import retained_bytes
from campaigns.adr_unified.adr_unified_plan import (
    _degree_variant, _sized_for_dofs, build_plan as build_unified_plan, fingerprint, memory_estimate, problem_class,
)

# (geometry, case) classes of the ladder. stress_main_trap and stress_main_cross are left out:
# every candidate (ASM, BJ, hp-MG, AMGX) stalled on them in the unified campaign and on the Blackwell.
CLASSES = (('square', 'trigonometric'), ('square', 'advection_dominated'), ('square', 'variable_velocity'),
           ('square', 'anisotropic'), ('annulus', 'cellular7_weak'), ('annulus', 'stress_main_orthogonal'))
DEGREES = (2, 6, 9)
MESH_LEVELS = ('L1', 'L2', 'L3', 'L4', 'L4b', 'L5')
# L6/L7 as in the unified plan (same ids); L8/L9 are new and beyond PARDISO's reach.
DOF_LEVELS = (('L6', 8_000_000), ('L7', 20_000_000), ('L8', 40_000_000), ('L9', 80_000_000))
# Lmax: per degree, the largest trace-DOF target (a multiple of LARGEST_STEP) whose system job
# passes the worker's device check on every class. The budget is the one the worker computed on
# the H200 NVL in the L7 tier (0.9 x free memory: 125.36 GiB, 51-l6-l7); the margin covers the
# annular meshes, whose triangle count the generator only approximates.
LARGEST_LEVEL = 'Lmax'
LARGEST_BUDGET_BYTES = int(.97*125.36*2**30)
LARGEST_STEP = 250_000

# Unified-campaign candidate ids per class: the fastest converging candidate of each family
# (median amortized time relative to the best on the same system, passed/attempted).
CLASS_CANDIDATES = {
    ('square', 'trigonometric'): ('ae88f9f8993c0220',   # asm_d24, 27/27
                                  'fdfae2409481602a',   # bj_pp_raw_fused_d12, 27/27
                                  'b13f235bf89bdf0d',   # amgx_pbicgstab_amg, 14/14
                                  '6222f36ecb62232b'),  # native_hp_standard, 27/27
    ('square', 'advection_dominated'): ('ae88f9f8993c0220',   # asm_d24, 29/29
                                        '1fd792e41e6e12f9',   # bj_pp_raw_fused_d12, 26/26, most wins
                                        'b13f235bf89bdf0d',   # amgx_pbicgstab_amg, 23/23
                                        '6222f36ecb62232b'),  # native_hp_standard, 29/29
    ('square', 'variable_velocity'): ('1b5c46c23218fefd',   # asm_pp_raw_fused_d18, 27/27
                                      '48c33996a420f31c',   # bj_pp_raw_fused_d18, 27/27
                                      'b13f235bf89bdf0d',   # amgx_pbicgstab_amg, 14/14
                                      '6222f36ecb62232b'),  # native_hp_standard, 27/27
    ('square', 'anisotropic'): ('d517fd34529cce69',   # asm_d48, 7/7, 6 wins
                                'd5474a265538eaef',   # amgx_pbicgstab_amg, 7/7
                                '6222f36ecb62232b'),  # native_hp_standard, 7/7
    ('annulus', 'cellular7_weak'): ('ae88f9f8993c0220',   # asm_d24, 38/38, only reliable candidate
                                    'b13f235bf89bdf0d'),  # amgx_pbicgstab_amg, 25/32, still converging at 33k-99k
    ('annulus', 'stress_main_orthogonal'): ('c0c1828753f52f50',   # asm_pp d96, 8/8
                                            '99ff59a75e236c3d'),  # bj_pp d96, 8/8
}
EXCLUDED = {
    'AMGX block-Jacobi / DILU / multicolor DILU smoothers (FGMRES, PBiCGStab)': '0 converged on every stress system',
    'AMGX AMG on cellular7_directional': '0/28',
    'AMGX DILU on cellular7_weak': '9/32 and 12/29, none above 8k triangles',
    'AMGX FGMRES+AMG on cellular7_weak': '17/32; PBiCGStab+AMG kept',
    'native_hp on cellular7_weak': '0 converged from 22,825 triangles up',
    'native_hp on stress_main_orthogonal': 'converges, but 92-110x slower than ASM+PP (18-20 s at 98k)',
    'pp (polynomial only)': 'stalls (trigonometric L3 p=6) and true_target slows its warm solves',
    'stress_main_trap, stress_main_cross': 'no candidate converges',
}

# H200 solver-kernel options (campaigns/solver_h200/README.md, slv-11..25). Operator and ASM/BJ
# 'interleaved' are bitwise equal to the campaign kernels; 'interleaved_f32' stores the ASM
# inverses in FP32 (FP64 accumulation; ASM+PP 98k p=6 repeated 2339 -> 557 ms with true_target).
# BJ keeps FP64 inverses: FP32 raised its warm iterations (delta=1e-3: 109 -> 146).
# 'graph' is the fastest hp-MG preset (2.0x robust, 1.44x standard at 98k p=6).
TUNED = {
    'asm_pp': dict(operator='interleaved', application='interleaved_f32', estimate_target='true_target'),
    'bj_pp': dict(operator='interleaved', application='interleaved', estimate_target='true_target'),
    'native_hp': dict(hp_mg_kernels='graph', operator_implementation='interleaved', estimate_target='true_target'),
    # Zero-copy face-BSR hand-off to AMGX (point 2): takes effect only with a pyamgx that has
    # attach_BSR (hybridge-venv-bsr against AMGX-build-cuda13.1-bsr); the production stack
    # uploads device-to-device and records bsr_borrowed=false.
    'amgx': dict(amgx_borrow_bsr=True),
}

# Raw-CUDA ADR tensor assembly by degree on sm_90 (Q1 adr-p*-n128 and asm-49/54 records):
# block 32 measured best at p<=4, where 'auto' picks 64 at p=3-4; 'auto' elsewhere gives
# 128 at p=5-7 and constant kinds, 256 at p=8 and 512 + non-compact + maxrregcount 128 at p=9
# for variable kinds. The explicit kind-6 mass inverse (p>=7) is the code default.
# schur-lu+mass caches the local factors for the cheapest reconstruction (p=9: 512 -> 61 ms).
RAW_ASSEMBLY_BLOCK = {1: 32, 2: 32, 3: 32, 4: 32}
RAW_ASSEMBLY = dict(family='raw_assembly', trace_basis='legacy-lagrange', matrix_format='bsr',
                    cache_local_factors='schur-lu+mass', timed_calls=3)


def raw_assembly_candidate(p):
    solver = dict(RAW_ASSEMBLY, block_size=RAW_ASSEMBLY_BLOCK.get(p, 'auto'))
    return dict(id=fingerprint(solver)[:16], aliases=['raw_assembly'], solver=solver)


def tuned_candidate(choice):
    """The unified candidate with the H200 kernel options; a new id, the old one kept for joins."""
    solver = deepcopy(choice['solver'])
    options = TUNED[solver['family']]
    if 'configuration' in solver:
        solver['configuration'].update({k: v for k, v in options.items() if k in ('operator', 'application')})
    solver.update({k: v for k, v in options.items() if k not in ('operator', 'application')})
    solver['tuned_face_dense'] = solver['family'] in ('asm_pp', 'bj_pp')
    return dict(id=fingerprint(solver)[:16], aliases=[*choice.get('aliases', []), 'h200_tuned'],
                base_candidate=choice['id'], solver=solver)


def system_device_need(system, candidates):
    """Device bytes the system job must fit, as ``adr_unified_worker`` checks them before assembly.

    The trace system, element blocks and verification data resident during ASM plus the largest
    candidate's planning solver bytes. The device-calibrated assembly, the other term of the
    worker's maximum, is far below this from L6 on.
    """
    triangles = system.get('triangles') or math.ceil(1.03*system['target_triangles'])
    solver = max(memory_estimate(system, c)['solver_device_bytes'] for c in candidates)
    return retained_bytes(triangles, system['problem']['p'])['during_asm'] + solver


def largest_dofs(bases, p, budget=LARGEST_BUDGET_BYTES):
    """The largest trace-DOF target whose system job fits ``budget`` for every (L5 base, solvers) pair."""
    def fits(steps):
        return all(system_device_need(_sized_for_dofs(base, p, LARGEST_LEVEL, steps*LARGEST_STEP, 'h200_tuned'),
                                      solvers) <= budget for base, solvers in bases)
    low, high = 1, 400
    if not fits(low) or fits(high):
        raise ValueError(f'p={p}: the largest fitting size is outside {LARGEST_STEP:,}-{high*LARGEST_STEP:,} DOFs')
    while high-low > 1:
        mid = (low+high)//2
        low, high = (mid, high) if fits(mid) else (low, mid)
    return low*LARGEST_STEP


def _class_systems(unified, key):
    return [s for s in unified['systems'] if problem_class(s)[:2] == key]


def _mesh_bases(systems):
    """(level, base system, degrees present) per archived or L5 mesh, as the unified h-p grid does."""
    by_mesh = defaultdict(list)
    for system in systems:
        if system['level'] in ('L6', 'L7') or system['origins'][0]['suite'] in ('hp_grid', 'lu_struggle'):
            continue
        mesh = system['mesh']['file'] if system.get('mesh') and 'file' in system['mesh'] else (
            'L5' if system['level'] == 'L5' else system['problem']['n'])
        by_mesh[mesh].append(system)
    bases = []
    for sources in by_mesh.values():
        level = next((s['level'] for s in sources if s['level'] != 'original'), None)
        if level is None and 'stress_parameters' in sources[0]['problem']:
            level = 'L4' if sources[0]['triangles'] < 120000 else 'L4b'
        if level is not None:
            bases.append((level, max(sources, key=lambda s: s['problem']['p']), sources))
    return bases


def build_plan(inventory, *, degrees=DEGREES, maxiter=2000, repeats=3, warmup=1,
               amgx_backend='cusparse_generic', dof_levels=DOF_LEVELS, classes=CLASSES, levels=None, largest=False):
    """The ladder; ``levels`` (e.g. a smoke or a first tier) keeps only those levels, ids unchanged.

    ``largest`` adds the ``Lmax`` level for every degree: its trace-DOF target is the same for
    all classes of the ladder (``CLASSES``), so a class subset keeps the sizes and ids.
    """
    unified = build_unified_plan(inventory, l5_triangles=400000, maxiter=maxiter, repeats=repeats, warmup=warmup,
                                 amgx_backend=amgx_backend, hp_grid=True, lu_levels=True)
    unified_ids = {s['id'] for s in unified['systems']}
    per_class = {}
    for key in CLASSES if largest else classes:
        members = _class_systems(unified, key)
        by_id = {c['id']: c for s in members for c in s['candidates']}
        missing = [c for c in CLASS_CANDIDATES[key] if c not in by_id]
        if missing:
            raise ValueError(f'{key}: candidates {missing} are not in the unified plan for this class')
        per_class[key] = (members, [tuned_candidate(by_id[c]) for c in CLASS_CANDIDATES[key]], _mesh_bases(members))
    largest_by_degree = {}
    if largest:
        l5_bases = [(next(base for level, base, _ in bases if level == 'L5'), candidates)
                    for _, candidates, bases in per_class.values()]
        largest_by_degree = {p: largest_dofs(l5_bases, p) for p in degrees}
    systems = []
    for key in classes:
        members, candidates, bases = per_class[key]
        chosen = {}
        for level, base, sources in bases:
            if level not in MESH_LEVELS:
                continue
            present = {s['problem']['p']: s for s in sources}
            for p in degrees:
                # An archived system at this degree keeps its own definition and id.
                system = deepcopy(present[p]) if p in present else _degree_variant(base, p, level, 'h200_tuned')
                system['level'] = level
                chosen[system['id']] = system
        l5 = next(base for level, base, _ in bases if level == 'L5')
        for level, dofs in dof_levels:
            for p in degrees:
                system = _sized_for_dofs(l5, p, level, dofs, 'h200_tuned')
                chosen[system['id']] = system
        for p, dofs in largest_by_degree.items():
            system = _sized_for_dofs(l5, p, LARGEST_LEVEL, dofs, 'h200_tuned')
            chosen[system['id']] = system
        for system in chosen.values():
            if levels is not None and system['level'] not in levels:
                continue
            system['candidates'] = [*deepcopy(candidates), raw_assembly_candidate(system['problem']['p'])]
            for choice in system['candidates']:
                choice['solver']['maxiter'] = maxiter
                choice['effective_solver_sha256'] = fingerprint(choice['solver'])
            system['shared_with_unified'] = system['id'] in unified_ids
            systems.append(system)
    if len({s['id'] for s in systems}) != len(systems):
        raise ValueError('Duplicate system identity in the H200 tuned ADR ladder')
    jobs = Counter(c['solver']['family'] for s in systems for c in s['candidates'])
    coverage = dict(total_systems=len(systems), solver_jobs=sum(jobs.values()), jobs_by_family=dict(jobs),
                    shared_with_unified=sum(s['shared_with_unified'] for s in systems),
                    systems_by_level=dict(Counter(s['level'] for s in systems)), degrees=sorted(degrees),
                    classes=[list(k) for k in classes])
    plan = dict(schema_version=1, campaign='adr_h200_tuned', inventory_sha256=unified['inventory_sha256'],
                protocol=unified['protocol'], amgx_backend=amgx_backend, l5_triangles=400000,
                tuned=TUNED, raw_assembly=dict(RAW_ASSEMBLY, block_by_degree=RAW_ASSEMBLY_BLOCK),
                excluded=EXCLUDED, systems=systems, coverage=coverage)
    if largest:
        plan['largest'] = dict(level=LARGEST_LEVEL, budget_bytes=LARGEST_BUDGET_BYTES, step=LARGEST_STEP,
                               trace_dofs_by_degree={str(p): d for p, d in sorted(largest_by_degree.items())})
    return plan
