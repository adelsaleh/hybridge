"""Launcher of the ADR selection cost study: plan, resume, provenance, completion marker.

Each task is one process of ``harness.py`` on one Hybridge tree, or one pytest session of this tree.
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
# pre: the revision the turbulence runs used before the fix. signature: device query cached. resolved: this tree.
# pre and signature are local worktrees of development revisions (see the local records).
TREES = dict(pre=REPO.parent/'hybridge-campaign-launch-fixed-20261102',
             signature=REPO.parent/'hybridge-campaign-article-retiming-20261102',
             resolved=REPO)
# Turbulence meshes: P2/p6 about 65,900 triangles, H150/p9 151,928.
CASES = dict(p6=dict(order=6, nx=182, diffusion='scalar', steps=500),
             p9=dict(order=9, nx=276, diffusion='scalar', steps=60),
             p6var=dict(order=6, nx=182, diffusion='variable', steps=200))
SMOKE = dict(order=3, nx=8, diffusion='variable', steps=6)
TESTS = ('tests/test_cuda_tuning.py', 'tests/test_adr_tuning_cuda.py', 'tests/test_adr_solver_reuse_cuda.py',
         'tests/test_adr_tensor_numba.py')
# Bounded subset of the 209 high-order cases, which alone exceed 45 minutes on one H200.
HIGH_ORDER = ('tests/test_adr_high_order_raw_cuda.py', '-k', 'legendre-modal-7')
# Both sides of the merge with `poisson-lu-responses`: tuning, ADR, Poisson launch, documentation and API contracts.
MERGE = ('tests/test_cuda_tuning.py', 'tests/test_adr_tuning_cuda.py', 'tests/test_adr_solver_reuse_cuda.py',
         'tests/test_adr_tensor_numba.py', 'tests/test_poisson_launch_resolution.py',
         'tests/test_documentation_structure.py', 'tests/test_solver_api_contract.py', 'tests/test_package_layering.py')
OUTPUT = REPO/'run_outputs'/'adr_selection_cost'
ARRAYS = Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_selection_cost'


def stages():
    """The fixed task list of every stage; arms alternate inside a repetition."""
    def run(case, arm, rep, **extra):
        return dict(id=f'{case}/{arm}-rep{rep}', kind='harness', arm=arm, case=case, rep=rep, **{**CASES.get(case, SMOKE), **extra})
    table = dict(
        smoke=[run('smoke', arm, 1) for arm in TREES],
        tests=[dict(id='pytest', kind='pytest'), dict(id='pytest-high-order', kind='pytest', selection='high_order')],
        attribution=[run('p6', 'pre', 0, spans=True, steps=1500), run('p9', 'pre', 0, spans=True),
                     run('p6', 'signature', 0, spans=True, steps=500), run('p6', 'resolved', 0, spans=True, steps=500)],
        timing=[run(case, arm, rep, save=True) for rep in (1, 2, 3) for case in CASES for arm in TREES],
        probe=[dict(id='probe', kind='probe', arm='pre', seconds=300.)],
    )
    table['first'] = table['tests'][:1] + table['attribution'] + table['tests'][1:]
    table['second'] = table['timing'] + table['probe']
    table['merge'] = [dict(id='pytest-merge', kind='pytest', selection='merge')]
    return table


def revision(path):
    """Commit and the sha256 of the tracked diff of a checkout."""
    git = lambda *a: subprocess.run(['git', '-C', str(path), *a], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(path), 'diff', 'HEAD'], capture_output=True).stdout
    import hashlib
    return dict(path=str(path), commit=git('rev-parse', 'HEAD'), diff_sha256=hashlib.sha256(diff).hexdigest())


def execute(item, out):
    """Run one task unless its marker exists; a failed task is recorded and the stage continues."""
    folder = out/item['id']
    done = folder/'DONE.json'
    if done.exists():
        return json.loads(done.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    tree = TREES[item.get('arm', 'resolved')]
    env = dict(os.environ, PYTHONPATH=str(tree), HYBRIDGE_CUDA_TUNING_CACHE_DIR=str(out/'cuda-tuning'/item.get('arm', 'tests')))
    if item['kind'] == 'pytest':
        command = [sys.executable, '-B', '-m', 'pytest', '-v', '--tb=short', '--durations=8', '-p', 'no:cacheprovider',
                   *dict(high_order=HIGH_ORDER, merge=MERGE).get(item.get('selection'), TESTS)]
        cwd = REPO
    else:
        command = [sys.executable, '-u', '-B', str(HERE/'harness.py'), '--out', str(folder), '--tree', str(tree)]
        cwd = folder
        if item['kind'] == 'probe':
            command += ['--probe-seconds', str(item['seconds'])]
        else:
            command += ['--order', str(item['order']), '--nx', str(item['nx']), '--steps', str(item['steps']),
                        '--diffusion', item['diffusion']]
            command += ['--spans'] if item.get('spans') else []
            if item.get('save'):
                ARRAYS.mkdir(parents=True, exist_ok=True)
                command += ['--save', str(ARRAYS/item['id'].replace('/', '-'))]
    start = time.perf_counter()
    with (folder/'task.log').open('w') as stream:
        code = subprocess.run(command, env=env, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT).returncode
    record = dict(item, returncode=code, seconds=time.perf_counter()-start, finished=time.strftime('%F %T'))
    log(f"{item['id']}: return code {code}, {record['seconds']:.1f} s")
    if code == 0:
        done.write_text(json.dumps(record, indent=1)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='smoke')
    parser.add_argument('--plan', action='store_true', help='print the tasks and exit')
    parser.add_argument('--out', type=Path, help='output directory (or HYBRIDGE_CAMPAIGN_OUT)')
    args = parser.parse_args()
    table = stages()
    if args.stage not in table:
        parser.error(f'unknown stage {args.stage}; choose from {sorted(table)}')
    tasks = table[args.stage]
    out = args.out or Path(os.environ.get('HYBRIDGE_CAMPAIGN_OUT') or OUTPUT)
    if args.plan:
        for item in tasks:
            print(json.dumps(item))
        print(f'# {len(tasks)} tasks; output {out}')
        return
    out.mkdir(parents=True, exist_ok=True)
    campaign = Campaign(out, exclusive=False)
    campaign.provenance(stage=args.stage, slurm_job=os.environ.get('SLURM_JOB_ID'), tasks=tasks,
                        trees={name: revision(tree) for name, tree in TREES.items()},
                        amgx=revision(REPO.parent/'AMGX'), pyamgx=revision(REPO.parent/'pyamgx'))
    records = [execute(item, out) for item in tasks]
    failed = [r['id'] for r in records if r.get('returncode') != 0]
    (out/f'COMPLETE_{args.stage}.json').write_text(json.dumps(dict(
        stage=args.stage, tasks=len(records), failed=failed, finished=time.strftime('%F %T')), indent=1)+'\n')
    log(f"stage {args.stage} finished; {len(records)} tasks, failed: {failed}")


if __name__ == '__main__':
    main()
