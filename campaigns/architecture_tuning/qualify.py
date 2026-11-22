#!/usr/bin/env python3
"""Bounded qualification of shared tuning and research-split compatibility."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.lifecycle import CampaignStop, stop_on_signals, terminate_group
from campaigns.common.protocol import host_provenance

STAGES = {
    'selection': ['tests/test_cuda_tuning.py', 'tests/test_cuda_variant_tuning.py',
                  'tests/test_cuda_tuned_batched_solve.py'],
    'transport': ['tests/test_advection_pipeline_tuning.py', 'tests/test_advection_tensor_auto.py',
                  'tests/test_split3_tuning.py', 'tests/test_transport_tuning_sample.py',
                  'tests/test_advection_tsle_bsr.py', 'tests/test_tsle_volume_gemm.py'],
    'adr': ['tests/test_adr_tuning.py', 'tests/test_adr_tuning_cuda.py',
            'tests/test_adr_high_order_raw_cuda.py::test_automatic_wide_launch_for_high_order_variable_tensors',
            'tests/test_adr_tensor_raw_cuda.py::test_reused_mass_factors_reproduce_fresh_assembly'],
    'solvers': ['tests/test_solver_kernel_tuning.py', 'tests/test_hp_mg_packed_kernels.py',
                'tests/test_legendre_face_bsr.py', 'tests/test_face_hp_mg_residual_restart.py',
                'tests/test_cupy_face_dense.py', 'tests/test_cupy_face_dense_coalesced.py',
                'tests/test_cupy_face_dense_interleaved.py', 'tests/test_cupy_additive_schwarz.py',
                'tests/test_cupy_polynomial.py'],
    'poisson': ['tests/test_poisson_high_order_cuda.py', '-k',
                'automatic or (direct_formats and 9 and bsr)'],
    'precision32': ['tests/test_architecture_tuning_fp32.py'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--stages', nargs='+', choices=tuple(STAGES), default=list(STAGES),
                        help='Run selected stages (default: all); use a distinct output per selection.')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get(
        'HYBRIDGE_CAMPAIGN_OUT', REPO / 'run_outputs' / 'architecture_tuning')))
    args = parser.parse_args()
    commands = {name: [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                       '--basetemp', str(args.output.resolve() / 'pytest' / name),
                       '--maxfail=3', *STAGES[name]] for name in args.stages}
    stage_environment = {name: dict(HYBRIDGE_PRECISION='float32' if name == 'precision32' else 'float64')
                         for name in args.stages}
    if args.plan:
        print(json.dumps(dict(output=str(args.output), gpu_count=1, commands=commands,
                              stage_environment=stage_environment), indent=2))
        return
    device = json.loads(subprocess.check_output([sys.executable, '-c',
        'import json; from hybridge.runtime.optional import require_cupy_device; '
        'from hybridge.runtime.cuda_tuning import device_signature; '
        'print(json.dumps(device_signature(require_cupy_device())))'], cwd=REPO, text=True))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pytest').mkdir(exist_ok=True)
    files = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '-z']).decode().split('\0')
    hashes = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in files
              if name.endswith(('.py', '.sh', '.sbatch')) and (REPO / name).is_file()}
    identity = hashlib.sha256(json.dumps(dict(files=hashes, commands=commands, device=device,
                                              stage_environment=stage_environment), sort_keys=True).encode()).hexdigest()
    marker = out / 'source-identity.txt'
    if marker.exists() and marker.read_text().strip() != identity:
        raise SystemExit('Sources, device, or commands changed; use a new output directory.')
    marker.write_text(identity+'\n')
    (out / 'sources.json').write_text(json.dumps(hashes, indent=2)+'\n')
    forks = {}
    for repo in ('AMGX', 'pyamgx'):
        forks[repo] = subprocess.run(['git', '-C', str(REPO.parent/repo), 'rev-parse', 'HEAD'],
                                     capture_output=True, text=True).stdout.strip()
    host_provenance(out, device=device, forks=forks, commands=commands, source_identity=identity,
                    stage_environment=stage_environment)
    env = dict(os.environ, PYTHONPATH=str(REPO), HYBRIDGE_CUDA_TUNING='auto',
               HYBRIDGE_CUDA_TUNING_CACHE_DIR=str(out / 'cache'))
    with stop_on_signals():
        for name, command in commands.items():
            if (out / (name+'.done')).exists():
                continue
            print('Starting '+name, flush=True)
            with (out / (name+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=REPO, env=dict(env, **stage_environment[name]),
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    status = child.wait()
                finally:
                    terminate_group(child)
            if status:
                raise SystemExit(f'{name} failed (exit {status}); see {out/name}.log')
            (out / (name+'.done')).touch()
    (out / 'COMPLETE').write_text('All requested architecture tuning qualification stages passed.\n')


if __name__ == '__main__':
    try:
        main()
    except CampaignStop:
        raise SystemExit(130)
