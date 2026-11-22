"""Does `dense_lu_max_rows` keep dense LU off a large coarsest AMG level, on every exit from coarsening?

AMGX's setup loop (`AMGX/src/amg.cu`) can finish in two ways. A level at `max_levels`, or at or
below `dense_lu_num_rows`, returns at the top of the loop, where the cap is checked. A level whose
next level is discarded for having fewer than `dense_lu_num_rows` rows breaks out at the bottom,
where it is not. These cases compare the exits on simple 2-D Laplacians with the article's Poisson
configuration, changing only those three settings.

A direct coarsest solve shows up as one Krylov iteration on a one-level hierarchy. On a two-level
hierarchy, identical residual histories show that the cap changed nothing. A non-converged case is
a recorded outcome: the question is which solver ran, not whether it converged.

    check_dense_lu_cap.py --plan        print the cases; touches neither CUDA nor files
    check_dense_lu_cap.py [--output D]  run on one GPU through scripts/gpu/run_cuda13.sh; resumes
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO/'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json'
ITERATION_LIMIT = 500
CAP_OF_THE_ARTICLE_CONFIG = 4096

# grid: points per direction. stop: dense_lu_num_rows. cap: dense_lu_max_rows, 0 for none.
# levels: max_levels, which forces the top-of-loop exit at that depth.
CASES = {
    # 6,400 rows. The next level is under the stop size and is discarded: bottom-of-loop exit, one level.
    'discard-nocap': dict(grid=80, stop=2048, cap=0),
    'discard-cap4096': dict(grid=80, stop=2048, cap=CAP_OF_THE_ARTICLE_CONFIG),
    'discard-cap6399': dict(grid=80, stop=2048, cap=6399),
    'discard-cap6400': dict(grid=80, stop=2048, cap=6400),
    # The same matrix and one level, leaving through max_levels: top-of-loop exit.
    'maxlevels-nocap': dict(grid=80, stop=2048, cap=0, levels=1),
    'maxlevels-cap4096': dict(grid=80, stop=2048, cap=CAP_OF_THE_ARTICLE_CONFIG, levels=1),
    'maxlevels-cap6399': dict(grid=80, stop=2048, cap=6399, levels=1),
    'maxlevels-cap6400': dict(grid=80, stop=2048, cap=6400, levels=1),
    # 3,025 rows, already at or below the stop size: top-of-loop exit without max_levels.
    'small-nocap': dict(grid=55, stop=4096, cap=0),
    'small-cap2048': dict(grid=55, stop=4096, cap=2048),
    # A real hierarchy whose coarsest level exceeds the cap. The repeat measures reproducibility.
    'deep-nocap': dict(grid=300, stop=4000, cap=0),
    'deep-nocap-repeat': dict(grid=300, stop=4000, cap=0),
    'deep-cap4096': dict(grid=300, stop=4000, cap=CAP_OF_THE_ARTICLE_CONFIG),
    'deep-maxlevels2-nocap': dict(grid=300, stop=4000, cap=0, levels=2),
    'deep-maxlevels2-cap4096': dict(grid=300, stop=4000, cap=CAP_OF_THE_ARTICLE_CONFIG, levels=2),
}


def run_case(name, directory):
    """One solve in its own process, so the native log holds exactly one hierarchy."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import cupy as cp
    import cupyx.scipy.sparse as device_sparse
    from hybridge.linalg.amgx.config import load_amgx_config
    from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver, amgx_device_memory_stats
    from campaigns.matk.common import amgx_library_provenance

    case = CASES[name]
    n = case['grid']
    line = sp.diags([-1., 2., -1.], [-1, 0, 1], shape=(n, n))
    identity = sp.identity(n)
    matrix = (sp.kron(line, identity)+sp.kron(identity, line)).tocsr()
    matrix.sort_indices()
    rhs = np.random.default_rng(7).normal(size=matrix.shape[0])
    rhs /= np.linalg.norm(rhs)
    reference = spla.spsolve(matrix.tocsc(), rhs)

    config, _ = load_amgx_config(CONFIG)
    config = copy.deepcopy(config)
    amg = config['solver']['preconditioner']
    amg.update(print_grid_stats=1, dense_lu_num_rows=case['stop'], dense_lu_max_rows=case['cap'])
    if 'levels' in case:
        amg['max_levels'] = case['levels']
    config['solver']['store_res_history'] = 1

    solver = PyAMGXCsrDeviceSolver(config=config, maxiter=ITERATION_LIMIT, verbose=3, residual_history_limit=None)
    device_matrix = device_sparse.csr_matrix(matrix)
    setup_seconds = solver.setup(device_matrix)
    after_setup = amgx_device_memory_stats()
    started = time.perf_counter()
    solution, info = solver.solve(cp.asarray(rhs))
    cp.cuda.Device().synchronize()
    solve_seconds = time.perf_counter()-started
    solution = cp.asnumpy(solution)
    library = amgx_library_provenance()
    solver.close()
    result = dict(case=case, rows=int(matrix.shape[0]), nonzeros=int(matrix.nnz), status=info['amgx_status'],
                  iterations=info['amgx_iterations'], residual_history=list(info['residual_history']),
                  relative_error=float(np.linalg.norm(solution-reference)/np.linalg.norm(reference)),
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds, memory_after_setup=after_setup, library=library)
    (directory/'result.json').write_text(json.dumps(result, indent=1, default=str)+'\n')


def summarize(root):
    """Join each result with the hierarchy AMGX printed; `parse_output` reads that native log."""
    from scripts.gpu.inspect_diffusion_amgx_hierarchy import parse_output
    rows = {}
    for name in CASES:
        directory = root/name
        if not (directory/'result.json').exists():
            rows[name] = dict(case=CASES[name], status='no result', returncode=_returncode(directory))
            continue
        result = json.loads((directory/'result.json').read_text())
        grid = parse_output((directory/'case.log').read_text(errors='replace'))
        rows[name] = {**result, **{k: grid.get(k) for k in ('amgx_levels', 'levels', 'coarsest_rows', 'operator_complexity')}}
    same = lambda a, b: rows[a].get('residual_history') == rows[b].get('residual_history')
    iterations = lambda name: rows[name].get('iterations')
    comparisons = dict(
        discard_cap4096_matches_nocap=same('discard-nocap', 'discard-cap4096'),
        discard_cap6399_matches_nocap=same('discard-nocap', 'discard-cap6399'),
        maxlevels_cap4096_differs_from_nocap=iterations('maxlevels-cap4096') != iterations('maxlevels-nocap'),
        maxlevels_cap6399_differs_from_nocap=iterations('maxlevels-cap6399') != iterations('maxlevels-nocap'),
        maxlevels_cap6400_matches_nocap=iterations('maxlevels-cap6400') == iterations('maxlevels-nocap'),
        small_cap2048_differs_from_nocap=iterations('small-cap2048') != iterations('small-nocap'),
        deep_repeat_is_reproducible=same('deep-nocap', 'deep-nocap-repeat'),
        deep_cap4096_matches_nocap=same('deep-nocap', 'deep-cap4096'),
        deep_coarsest_rows_exceed_cap=(rows['deep-nocap'].get('coarsest_rows') or 0) > CAP_OF_THE_ARTICLE_CONFIG,
        deep_maxlevels2_cap4096_differs=not same('deep-maxlevels2-nocap', 'deep-maxlevels2-cap4096'))
    return dict(comparisons=comparisons, cases=rows)


def _returncode(directory):
    path = directory/'returncode'
    return int(path.read_text()) if path.exists() else None


def print_table(summary):
    print(f"{'case':<26}{'levels':>7}{'coarsest':>10}{'iterations':>11}  {'status':<14}{'rel. error':>11}{'solve s':>9}")
    for name, row in summary['cases'].items():
        error = row.get('relative_error')
        print(f"{name:<26}{str(row.get('amgx_levels')):>7}{str(row.get('coarsest_rows')):>10}{str(row.get('iterations')):>11}  "
              f"{str(row.get('status')):<14}{'--' if error is None else f'{error:.1e}':>11}{row.get('solve_seconds', float('nan')):>9.3f}")
    print(json.dumps(summary['comparisons'], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--case', choices=CASES, help='Internal: run one case in this process.')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get(
        'HYBRIDGE_CAMPAIGN_OUT', REPO/'run_outputs/fork_integration/dense_lu_cap')))
    args = parser.parse_args()
    root = args.output.resolve()
    if args.plan:
        print(json.dumps(dict(output=str(root), config=str(CONFIG), cases=CASES), indent=1))
        return 0
    if args.case:
        run_case(args.case, root/args.case)
        return 0

    from campaigns.common.protocol import host_provenance
    host_provenance(root, slurm_gpus=os.environ.get('SLURM_JOB_GPUS'), config=str(CONFIG), cases=CASES)
    for name in CASES:
        directory = root/name
        if (directory/'result.json').exists():
            print(f'RESUME {name}', flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'case.log').open('w') as log:
            code = subprocess.run([sys.executable, '-u', '-B', __file__, '--case', name, '--output', str(root)],
                                  cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        (directory/'returncode').write_text(str(code))
        print(f'END {name} returncode={code}', flush=True)
    summary = summarize(root)
    (root/'summary.json').write_text(json.dumps(summary, indent=1, default=str)+'\n')
    print_table(summary)
    (root/'complete.json').write_text(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'),
        finished=datetime.now(timezone.utc).isoformat()))+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
