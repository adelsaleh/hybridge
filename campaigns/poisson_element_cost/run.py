"""Launcher of the Poisson per-element cost investigation: plan, resume, provenance, completion marker.

A stage is a fixed task list. Each task is one process of ``harness.py`` on one article row, with one
Hybridge tree (the frozen baseline or the uncommitted experimental worktree) and its switches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from campaigns.common.protocol import Campaign, log  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = REPO.parent/'guiding-center-cg-hdg'
TREES = dict(fixed=REPO.parent/'hybridge-campaign-launch-fixed-20261102',
             experiment=REPO.parent/'hybridge-poisson-element-cost-20261102',
             revised=REPO.parent/'hybridge-campaign-article-retiming-20261102',
             factors=REPO.parent/'hybridge-poisson-local-factors-20261102',
             promoted=REPO.parent/'hybridge-poisson-lu-responses-20261102')
ROW_SMALL = 'wave_ark3-p4-h0.08-compatible'
ROW_MEDIUM = 'wave_ark3-p2-h0.02-compatible'
ROWS_HIGH = ('wave_bdf2-p7-q7-compatible', 'wave_bdf2-p8-q8-compatible', 'wave_bdf2-p9-q9-compatible')
ROWS_LOW = ('wave_ark3-p4-h0.005-compatible', 'wave_ark3-p2-h0.005-compatible')
HOST_ALL = 'signature,resolved,nocheck'
# Local-factor question: rows with their triangle counts, and the rows that run the p+1 recovery.
ROWS_FACTORS = {ROW_SMALL: (4, 1474), ROW_MEDIUM: (2, 23254), 'wave_ark3-p6-h0.02-compatible': (6, 23254),
                ROWS_LOW[0]: (4, 369790), 'wave_ark3-p2-h0.0025-compatible': (2, 1478818),
                ROWS_HIGH[0]: (7, 92552), ROWS_HIGH[1]: (8, 92552), ROWS_HIGH[2]: (9, 92552),
                'wave_bdf2-p9-q9-raw': (9, 92552), 'vortex-p6-h0.025': (6, None)}
OUTPUT = REPO/'run_outputs'/'poisson_element_cost'


def task(identifier, row, tree='fixed', mode='real', steps=None, host='', kernel='', options='', variants=None, overrides=None,
         local='', primal_bytes=None):
    return dict(id=identifier, row=row, tree=tree, mode=mode, steps=steps, host=host, kernel=kernel,
                options=options, variants=variants, overrides=overrides or {}, local=local, primal_bytes=primal_bytes)


def repeated_arms(row, arms, repeats, steps=None):
    """Alternate arms inside each repetition, so that slow drift falls equally on both."""
    return [task(f'{row}/{name}-rep{rep}', row, steps=steps, **arm)
            for rep in range(1, repeats+1) for name, arm in arms.items()]


def kernel_variants(kind='full'):
    """Variants replayed on the captured arrays. The first one is the parity reference."""
    tuned = ['tuned']
    sizes = [32, 64, 128, 256]
    base = [dict(name='base-tuned', threads=tuned),
            dict(name='base-clock', flags=dict(PHASE_CLOCK=1), threads=tuned)]
    if kind == 'smoke':
        return base + [dict(name='warp', flags=dict(WARP_SOLVE=1), threads=tuned)]
    if kind == 'response-check':
        # Stored responses against a full local solve with the same Cholesky factor and trace.
        return [dict(name='responses', threads=[64], phases=[2], save=True),
                dict(name='cholesky-solve', flags=dict(NO_RESPONSE=1), threads=[64], phases=[2], use_moments=True, save=True)]
    if kind == 'lu-save':
        return [dict(name='lu', threads=[64], phases=[2], save=True)]
    if kind in ('round2-lu', 'round2-cholesky'):
        rows = base + [dict(name='base-sweep', threads=sizes),
                       dict(name='base-clock-sweep', flags=dict(PHASE_CLOCK=1), threads=sizes),
                       dict(name='factor-load-only', flags=dict(STOP=2), threads=[64, 128, 256])]
        low = [32, 64, 128]
        sets = [('warp', dict(WARP_SOLVE=1)),
                ('warp-skip', dict(WARP_SOLVE=1, SKIP_INTERIOR=1)),
                ('warp-direct', dict(WARP_SOLVE=1, DIRECT_FACTORS=1)),
                ('warp-direct-skip', dict(WARP_SOLVE=1, DIRECT_FACTORS=1, SKIP_INTERIOR=1)),
                ('warp-wide4-skip', dict(WARP_SOLVE=1, LOAD_WIDE=4, SKIP_INTERIOR=1)),
                ('inverse', dict(INVERSE=1)),
                ('inverse-skip', dict(INVERSE=1, SKIP_INTERIOR=1))]
        for name, flags in sets:
            rows.append(dict(name=name, flags=flags, threads=sizes))
            rows.append(dict(name=name+'-clock', flags=dict(flags, PHASE_CLOCK=1), threads=low))
        for regs in (64, 80, 96):
            for name, flags in sets[1:]:
                rows.append(dict(name=f'maxrreg{regs}-{name}', options=[f'--maxrregcount={regs}'], flags=flags, threads=low))
        if kind == 'round2-lu':
            # The L1 threshold: fixed 64-thread block, total shared memory raised through the padding.
            for pad in (20, 22, 24, 26, 28):
                rows.append(dict(name=f'pad{pad}k-t64', pad_to=pad*1024, threads=[64], flags=dict(PHASE_CLOCK=1)))
            for pad in (6, 10, 15, 20, 22, 24, 26, 28):
                rows.append(dict(name=f'warp-direct-pad{pad}k-t64', pad_to=pad*1024, threads=[64],
                                 flags=dict(WARP_SOLVE=1, DIRECT_FACTORS=1, PHASE_CLOCK=1)))
        return rows
    if kind == 'cholesky':
        # The existing compact policy, whose reconstruction needs no factors or local solve.
        return base + [dict(name='base-sweep', threads=sizes),
                       dict(name='base-clock-sweep', flags=dict(PHASE_CLOCK=1), threads=sizes),
                       dict(name='skip', flags=dict(SKIP_INTERIOR=1), threads=sizes),
                       dict(name='contract-q', flags=dict(CONTRACT_Q=1), threads=sizes),
                       dict(name='ptxas-O1', options=['--ptxas-options=-O1'], threads=sizes)]
    rows = base + [dict(name='base-sweep', threads=sizes),
                   dict(name='base-clock-sweep', flags=dict(PHASE_CLOCK=1), threads=sizes),
                   dict(name='base-retain', retain=True, threads=sizes),
                   dict(name='factor-load-only', flags=dict(STOP=2), threads=[64, 128, 256]),
                   dict(name='factor-load-only-wide8', flags=dict(STOP=2, LOAD_WIDE=8), threads=[64, 128, 256])]
    single = (('warp', dict(WARP_SOLVE=1)),
              ('skip', dict(SKIP_INTERIOR=1)),
              ('contract-q', dict(CONTRACT_Q=1)),
              ('contract-coupling', dict(CONTRACT_COUPLING=1)),
              ('wide4', dict(LOAD_WIDE=4)),
              ('wide8', dict(LOAD_WIDE=8)))
    combined = (('warp-direct', dict(WARP_SOLVE=1, DIRECT_FACTORS=1)),
                ('warp-wide8', dict(WARP_SOLVE=1, LOAD_WIDE=8)),
                ('warp-skip', dict(WARP_SOLVE=1, SKIP_INTERIOR=1)),
                ('warp-skip-cq', dict(WARP_SOLVE=1, SKIP_INTERIOR=1, CONTRACT_Q=1)),
                ('warp-skip-cq-cc', dict(WARP_SOLVE=1, SKIP_INTERIOR=1, CONTRACT_Q=1, CONTRACT_COUPLING=1)),
                ('warp-skip-cq-cc-wide8', dict(WARP_SOLVE=1, SKIP_INTERIOR=1, CONTRACT_Q=1, CONTRACT_COUPLING=1, LOAD_WIDE=8)),
                ('warp-direct-skip-cq-cc', dict(WARP_SOLVE=1, DIRECT_FACTORS=1, SKIP_INTERIOR=1, CONTRACT_Q=1, CONTRACT_COUPLING=1)))
    for name, flags in single + combined:
        rows.append(dict(name=name, flags=flags, threads=sizes))
        rows.append(dict(name=name+'-clock', flags=dict(flags, PHASE_CLOCK=1), threads=[64, 128]))
    for name, options in (('ptxas-O1', ['--ptxas-options=-O1']), ('maxrreg80', ['--maxrregcount=80'])):
        rows.append(dict(name=name, options=options, threads=sizes))
        rows.append(dict(name=name+'-warp-skip-cq-cc', options=options, threads=sizes,
                         flags=dict(WARP_SOLVE=1, SKIP_INTERIOR=1, CONTRACT_Q=1, CONTRACT_COUPLING=1)))
        rows.append(dict(name=name+'-clock', options=options, flags=dict(PHASE_CLOCK=1), threads=[64, 128]))
    # Resident blocks per SM forced down at a fixed block size: contention versus per-block latency.
    for threads in (64, 128):
        for pad in (40*1024, 60*1024, 80*1024, 120*1024, 228*1024):
            rows.append(dict(name=f'pad{pad//1024}k-t{threads}', pad_to=pad, threads=[threads], flags=dict(PHASE_CLOCK=1)))
    return rows


def stages():
    return dict(
        smoke=[task('spans-fixed', ROW_SMALL, mode='spans', steps=8),
               task('kernel-experiment', ROW_SMALL, tree='experiment', mode='kernel', steps=8, variants='smoke')],
        attribution=[task(f'spans-{row}-rep{rep}', row, mode='spans') for row in (ROW_SMALL, ROW_MEDIUM) for rep in (1, 2)]
        + [task(f'cprofile-{row}', row, mode='cprofile') for row in (ROW_SMALL, ROW_MEDIUM)],
        kernels_p9=[task(f'kernel-{ROWS_HIGH[2]}', ROWS_HIGH[2], tree='experiment', mode='kernel', variants='full'),
                    task(f'kernel-cholesky-{ROWS_HIGH[2]}', ROWS_HIGH[2], tree='experiment', mode='kernel', variants='cholesky',
                         overrides=dict(poisson_cache_local_factors='schur-cholesky'))],
        kernels_p78=[task(f'kernel-{row}', row, tree='experiment', mode='kernel', variants='full') for row in ROWS_HIGH[:2]],
        response_check=[task(f'response-check-{row}', row, tree='experiment', mode='kernel', variants='response-check',
                             overrides=dict(poisson_cache_local_factors='schur-cholesky')) for row in ROWS_HIGH]
        + [task(f'lu-save-{row}', row, tree='experiment', mode='kernel', variants='lu-save') for row in ROWS_HIGH],
        round2=[task(f'kernel-{policy}-{row}', row, tree='experiment', mode='kernel', variants=f'round2-{policy}',
                     overrides=dict(poisson_cache_local_factors=f'schur-{policy}') if policy == 'cholesky' else None)
                for row in ROWS_HIGH[::-1] for policy in ('lu', 'cholesky')],
    )


def real_arm_stages():
    """Real-run arms. Kernel switches were fixed from the probes of ``kernels_p9`` and ``round2``."""
    cholesky = dict(poisson_cache_local_factors='schur-cholesky')
    warp = 'WARP_SOLVE,DIRECT_FACTORS,SKIP_INTERIOR'
    inverse = 'INVERSE,SKIP_INTERIOR'
    host = dict(fixed=dict(tree='fixed'),
                default=dict(tree='experiment'),
                signature=dict(tree='experiment', host='signature'),
                hostall=dict(tree='experiment', host=HOST_ALL))
    rows = []
    for row in (ROW_SMALL, ROW_MEDIUM):
        rows += repeated_arms(row, host, 3)
    boundary = dict(fixed=dict(tree='fixed'),
                    hostall=dict(tree='experiment', host=HOST_ALL),
                    hostboundary=dict(tree='experiment', host=HOST_ALL+',boundary'))
    device = dict(fixed=dict(tree='fixed'),
                  default=dict(tree='experiment'),
                  hostall=dict(tree='experiment', host=HOST_ALL),
                  luwarp=dict(tree='experiment', kernel=warp),
                  luinverse=dict(tree='experiment', kernel=inverse),
                  chol=dict(tree='experiment', overrides=cholesky),
                  cholwarp=dict(tree='experiment', kernel=warp, overrides=cholesky),
                  cholinverse=dict(tree='experiment', kernel=inverse, overrides=cholesky),
                  luwarphost=dict(tree='experiment', host=HOST_ALL, kernel=warp),
                  cholwarphost=dict(tree='experiment', host=HOST_ALL, kernel=warp, overrides=cholesky))
    high = []
    for row in ROWS_HIGH:
        high += repeated_arms(row, device, 3)
    device2 = dict(fixed=dict(tree='fixed'),
                   luinversehost=dict(tree='experiment', host=HOST_ALL, kernel=inverse),
                   luwarphost=dict(tree='experiment', host=HOST_ALL, kernel=warp))
    second = []
    for row in ROWS_HIGH:
        second += repeated_arms(row, device2, 3)
    low_arms = dict(fixed=dict(tree='fixed'),
                    hostall=dict(tree='experiment', host=HOST_ALL),
                    luwarphost=dict(tree='experiment', host=HOST_ALL, kernel=warp),
                    luinversehost=dict(tree='experiment', host=HOST_ALL, kernel=inverse),
                    cholhost=dict(tree='experiment', host=HOST_ALL, overrides=cholesky),
                    cholwarphost=dict(tree='experiment', host=HOST_ALL, kernel=warp, overrides=cholesky))
    low = []
    for row in ROWS_LOW:
        low += repeated_arms(row, low_arms, 2)
    return dict(host=rows, boundary=repeated_arms(ROW_SMALL, boundary, 3), device=high, device2=second, lowdegree=low)


def primal_inverse_bytes(p, triangles):
    """Device limit that admits the recovery's inverse operator but not its larger P,Q operator."""
    post = (p+2)*(p+3)//2
    return int(1.001*triangles*post*(post+1)*8)


def factor_arms(p, triangles, compatible):
    """One tree and revision: plain cached LU, the two switches, no cache, and the recovery's three paths."""
    arms = dict(control=dict(tree='revised'), base=dict(tree='factors'),
                responses=dict(tree='factors', local='RESPONSES'), inverse=dict(tree='factors', local='INVERSE'),
                both=dict(tree='factors', local='RESPONSES,INVERSE'),
                nocache=dict(tree='factors', overrides=dict(poisson_cache_local_factors='none')))
    if compatible and triangles:
        arms.update(primalinverse=dict(tree='factors', primal_bytes=primal_inverse_bytes(p, triangles)),
                    primalsolve=dict(tree='factors', primal_bytes=1))
    return arms


def factor_stages():
    def rows(selected, repeats, steps=None):
        out = []
        for row in selected:
            p, triangles = ROWS_FACTORS[row]
            out += repeated_arms(row, factor_arms(p, triangles, row.endswith('-compatible')), repeats, steps=steps)
        return out
    tests = task('tree-tests', ROW_SMALL, tree='factors', mode='pytest')
    final = dict(control=dict(tree='revised'),
                 promoted=dict(tree='promoted', overrides=dict(poisson_cache_local_factors='schur-lu+responses')))
    selected = (ROWS_LOW[0], 'wave_ark3-p2-h0.0025-compatible', 'wave_ark3-p6-h0.02-compatible', 'wave_ark3-p6-h0.02-qpplus1',
                'wave_bdf2-p6-q7-raw', *ROWS_HIGH, 'wave_bdf2-p9-q9-raw')
    confirmed = [task('tree-tests', ROW_SMALL, tree='promoted', mode='pytest')]
    for row in selected:
        confirmed += repeated_arms(row, final, 3)
    confirmed += repeated_arms('vortex-p6-h0.025', final, 3, steps=40)
    return dict(factors_smoke=[tests]+rows((ROW_SMALL, ROWS_HIGH[2]), 1, steps=8), factors=[tests]+rows(ROWS_FACTORS, 3),
                factors_final=confirmed)


def all_stages():
    return dict(stages(), **real_arm_stages(), **factor_stages())


def execute(item, out, cache):
    target = out/item['id']
    target.mkdir(parents=True, exist_ok=True)
    marker = target/'DONE.json'
    if marker.exists():
        log(f"resume: {item['id']} already complete")
        return json.loads(marker.read_text())
    tree = TREES[item['tree']]
    env = dict(os.environ, PYTHONPATH=f'{RESEARCH}:{tree}', HYBRIDGE_REPO=str(tree),
               HYBRIDGE_CUDA_TUNING_CACHE_DIR=str(cache),
               HYBRIDGE_POISSON_HOST_FLAGS=item['host'], HYBRIDGE_POISSON_KERNEL_FLAGS=item['kernel'],
               HYBRIDGE_POISSON_LOCAL_FLAGS=item.get('local', ''),
               HYBRIDGE_POISSON_KERNEL_OPTIONS=item['options'],
               HYBRIDGE_POISSON_ELEMENT_COST_OVERRIDES=json.dumps(item.get('overrides') or {}))
    if item.get('primal_bytes'):
        env['HYBRIDGE_PRIMAL_CACHE_MAX_BYTES'] = str(item['primal_bytes'])
    command = [sys.executable, '-u', '-B', str(HERE/'harness.py'), '--row', item['row'], '--out', str(target), '--mode', item['mode']]
    if item['mode'] == 'pytest':
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_poisson_launch_resolution.py',
                   'tests/test_poisson_high_order_cuda.py', 'tests/test_cuda_tuning.py', 'tests/test_solver_api_contract.py',
                   'tests/test_diffusion_reaction_assembly_parity.py', 'tests/test_diffusion_recovery_cache.py']
    if item['steps']:
        command += ['--steps', str(item['steps'])]
    if item['variants']:
        variants = target/'variants.json'
        variants.write_text(json.dumps(kernel_variants(item['variants']), indent=1))
        command += ['--variants', str(variants)]
    log(f"start {item['id']} tree={item['tree']} mode={item['mode']} host={item['host']!r} kernel={item['kernel']!r} options={item['options']!r}")
    started = time.time()
    with (target/'harness.log').open('a') as stream:
        code = subprocess.run(command, cwd=tree if item['mode'] == 'pytest' else RESEARCH, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    record = dict(item, returncode=code, seconds=time.time()-started, finished=time.strftime('%F %T'))
    log(f"finish {item['id']} rc={code} in {record['seconds']:.0f} s")
    if code == 0:
        marker.write_text(json.dumps(record, indent=1)+'\n')
    else:
        (target/'FAILED.json').write_text(json.dumps(record, indent=1)+'\n')
    return record


def revision(path):
    return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def tree_state(tree):
    diff = subprocess.run(['git', '-C', str(tree), 'diff', 'HEAD'], capture_output=True, text=True).stdout
    import hashlib
    return dict(path=str(tree), commit=revision(tree), diff_sha256=hashlib.sha256(diff.encode()).hexdigest(), diff_bytes=len(diff))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='smoke')
    parser.add_argument('--plan', action='store_true', help='print the tasks and exit')
    parser.add_argument('--out', type=Path, help='output directory (or HYBRIDGE_CAMPAIGN_OUT)')
    args = parser.parse_args()
    table = all_stages()
    if args.stage not in table:
        parser.error(f'unknown stage {args.stage}; choose from {sorted(table)}')
    tasks = table[args.stage]
    if args.plan:
        for item in tasks:
            print(json.dumps(item))
        print(f'# {len(tasks)} tasks; output {os.environ.get("HYBRIDGE_CAMPAIGN_OUT") or args.out or OUTPUT/args.stage}')
        return
    out = args.out or Path(os.environ.get('HYBRIDGE_CAMPAIGN_OUT') or OUTPUT/args.stage)
    out.mkdir(parents=True, exist_ok=True)
    campaign = Campaign(out, exclusive=False)
    campaign.provenance(stage=args.stage, slurm_job=os.environ.get('SLURM_JOB_ID'),
                        trees={name: tree_state(tree) for name, tree in TREES.items()},
                        amgx=revision(REPO.parent/'AMGX'), pyamgx=revision(REPO.parent/'pyamgx'),
                        research=revision(RESEARCH), tasks=tasks)
    cache = out/'cuda-tuning'
    records = [execute(item, out, cache) for item in tasks]
    failed = [r['id'] for r in records if r.get('returncode') != 0]
    (out/f'COMPLETE_{args.stage}.json').write_text(json.dumps(dict(
        stage=args.stage, tasks=len(records), failed=failed, finished=time.strftime('%F %T')), indent=1)+'\n')
    log(f"stage {args.stage} finished; {len(records)} tasks, failed: {failed}")


if __name__ == '__main__':
    main()
