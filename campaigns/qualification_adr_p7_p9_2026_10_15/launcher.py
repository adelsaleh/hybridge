#!/usr/bin/env python3
"""Detached ADR kernel qualification; resume successful stages from exact sources."""

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_output = _campaign_data_directory('hybridge/run_outputs/qualification/adr_p7_p9_2026_10_15')

from datetime import datetime, timezone
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

STUDY = _campaign_output
REPOSITORY = Path(str(_campaign_workspace_path('hybridge')))
FROZEN_PROVENANCE = Path(os.environ.get(
    'HYBRIDGE_QUALIFICATION_PROVENANCE',
    REPOSITORY.parent / 'guiding-center-cg-hdg' / 'provenance' /
    'remaining_hybridge_outputs' / 'run_outputs' / 'qualification' /
    'adr_p7_p9_2026_10_15',
)).expanduser().resolve()
FROZEN_MANIFEST = FROZEN_PROVENANCE / 'manifest.json'
SOURCE = FROZEN_PROVENANCE / 'source'
PYTHON = str(REPOSITORY / '.venv/bin/python')
SANITIZER = '/usr/local/cuda-13.0/bin/compute-sanitizer'
os.environ.update(PYTHONUNBUFFERED='1', PYTHONPATH=str(SOURCE),
                  HYBRIDGE_PRECISION='float64', OPENBLAS_NUM_THREADS='1',
                  NUMBA_NUM_THREADS='16', OMP_NUM_THREADS='16')
sys.path.insert(0, str(SOURCE))


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'{now()} {message}', flush=True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_stage(name, nodes, *, sanitizer=None, completed_nodes=()):
    """Run one immutable test stage, retaining its log, XML and success marker."""
    manifest_bytes = FROZEN_MANIFEST.read_bytes()
    xml = STUDY / f'{name}.xml'
    command = [PYTHON, '-u', '-m', 'pytest', '-vv', '-s', '--maxfail=1',
               '--durations=10', f'--junitxml={xml}', *nodes]
    if sanitizer:
        command = [SANITIZER, '--tool', sanitizer, '--error-exitcode', '99',
                   '--target-processes', 'all', *command]
    fingerprint = hashlib.sha256(manifest_bytes + json.dumps(command).encode()).hexdigest()
    marker = STUDY / f'{name}.complete.json'
    if marker.exists() and json.loads(marker.read_text()).get('fingerprint') == fingerprint:
        log(f'SKIP completed stage {name}')
        return
    reused = sorted(set(completed_nodes))
    logical_command = list(command)
    if reused:
        # The saved collection and sources are immutable. Resume the remaining
        # node IDs, retaining the original scope/fingerprint and prior evidence.
        if sanitizer or nodes != ['tests/test_adr_tensor_raw_cuda.py']:
            raise ValueError('partial-node resume is scoped to the interrupted parity stage')
        expected = {line.strip() for line in (STUDY / 'collection.txt').read_text().splitlines()
                    if line.startswith('tests/test_adr_tensor_raw_cuda.py::')}
        if len(expected) != 294 or not set(reused) <= expected:
            raise RuntimeError('saved parity passes do not match the frozen collection')
        remaining = sorted(expected - set(reused))
        if not remaining:
            raise RuntimeError('all nodes already passed; consolidate their evidence before resuming')
        command = [*command[:-len(nodes)], *remaining]
        log(f'RESUME {name}: reuse {len(reused)} passed nodes; run {len(remaining)} remaining nodes')
    record = dict(stage=name, state='running', pid=os.getpid(), pgid=os.getpgrp(),
                  started=now(), command=command, fingerprint=fingerprint,
                  logical_command=logical_command, reused_nodes=reused,
                  prior_passes=len(reused), gpu_reservation=False)
    write_json(STUDY / 'status.json', record)
    log(f'START {name}: {shlex.join(command)}')
    start = time.monotonic()
    with (STUDY / f'{name}.log').open('a', buffering=1) as stream:
        stream.write(f'\n{now()} {shlex.join(command)}\n')
        process = subprocess.Popen(command, cwd=SOURCE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        record['child_pid'] = process.pid
        write_json(STUDY / 'status.json', record)
        for line in process.stdout:
            stream.write(line)
            print(line, end='', flush=True)
        returncode = process.wait()
    record.update(returncode=returncode, finished=now(), seconds=time.monotonic()-start)
    if returncode == 0:
        suites = ET.parse(xml).getroot().findall('.//testsuite')
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        if counts['tests'] == 0 or any(counts[key] for key in ('failures', 'errors', 'skipped')):
            record['qualification_error'] = 'empty, failed or skipped qualification tests'
            returncode = 1
        record['current_invocation_counts'] = dict(counts)
        counts['tests'] += len(reused)
        record['counts'] = counts
        if reused and counts['tests'] != 294:
            record['qualification_error'] = 'resumed parity stage did not cover all 294 nodes'
            returncode = 1
    record['state'] = 'passed' if returncode == 0 else 'failed'
    write_json(STUDY / f'{name}.result.json', record)
    if returncode:
        raise RuntimeError(f'{name} failed; see {name}.log and {name}.result.json')
    write_json(marker, record)
    log(f'PASS {name}: {record["counts"]}; {record["seconds"]:.1f}s')


def main():
    """Verify frozen code and run gates concurrently, without a GPU reservation."""
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    for name, digest in manifest['source']['files_sha256'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'frozen qualification source changed: {name}')
    import hybridge
    assert Path(hybridge.__file__).resolve().is_relative_to(SOURCE), hybridge.__file__
    log(f'Using frozen source {hybridge.__file__}; baseline {manifest["baseline"]}')
    run_stage('host', [
        'tests/test_raw_cuda_policy.py',
        'tests/test_adr_tensor_raw_cuda.py::test_workspace_specializations',
        'tests/test_adr_tensor_raw_cuda.py::test_workspace_limit_is_a_clear_configuration_error',
        'tests/test_adr_high_order_raw_cuda.py::test_high_order_resource_planning',
        'tests/test_adr_high_order_raw_cuda.py::test_high_order_excessive_quadrature_still_has_a_bounded_error',
        'tests/test_package_layering.py', 'tests/test_backend_capabilities.py',
    ])
    write_json(STUDY / 'status.json', dict(state='starting-gpu-tests', pid=os.getpid(),
                                          pgid=os.getpgrp(), updated=now(), gpu_reservation=False))
    log('GPU tests run concurrently without taking the GPU lock, as requested by the user')
    with nullcontext():
        environment = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv'], text=True)
        (STUDY / 'device.txt').write_text(environment)
        run_stage('smoke', [
            'tests/test_adr_tensor_raw_cuda.py::test_tensor_assembly_all_shapes[variable-general-legendre-modal-6]',
            'tests/test_adr_high_order_raw_cuda.py::test_high_order_cupy_reference_and_cache_policies[variable-general-legendre-modal-9]',
        ])
        completed = set()
        interruption = STUDY / 'interruption.json'
        if interruption.exists():
            completed.update(json.loads(interruption.read_text())['passed_nodes'])
            previous_log = STUDY / 'numpy_numba_parity.log'
            completed.update(line.split(' PASSED', 1)[0] for line in previous_log.read_text().splitlines()
                             if line.startswith('tests/test_adr_tensor_raw_cuda.py::') and line.endswith(' PASSED'))
        run_stage('numpy_numba_parity', ['tests/test_adr_tensor_raw_cuda.py'], completed_nodes=completed)
        run_stage('cupy_and_resource_parity', ['tests/test_adr_high_order_raw_cuda.py'])
        run_stage('native_csr_bsr', [
            'tests/test_adr_tensor_solver_cuda.py::test_native_tensor_solve_and_device_reconstruction',
        ])
        stress = [
            'tests/test_adr_high_order_raw_cuda.py::test_high_order_external_mass_chunks_and_reuse'
            f'[{kind}-128-csr-{cache}]'
            for kind in (4, 5, 6) for cache in ('none', 'schur-lu+mass')
        ] + [
            'tests/test_adr_high_order_raw_cuda.py::test_mass_lu_pivots_cross_four_warp_row_groups[128]',
            'tests/test_adr_high_order_raw_cuda.py::test_high_order_cupy_reference_and_cache_policies[variable-general-legendre-modal-7]',
        ]
        run_stage('memcheck', stress, sanitizer='memcheck')
        run_stage('racecheck', stress, sanitizer='racecheck')


if __name__ == '__main__':
    # An additional lock prevents accidentally duplicating this qualification.
    with (STUDY / 'launcher.lock').open('a') as instance:
        fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (STUDY / 'pgid').write_text(str(os.getpgrp()) + '\n')
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        terminal = dict(pid=os.getpid(), pgid=os.getpgrp(), started=now())
        try:
            main()
        except BaseException as error:
            traceback.print_exc()
            terminal.update(state='failed', error=str(error), finished=now())
            write_json(STUDY / 'completion.json', terminal)
            write_json(STUDY / 'status.json', terminal)
            raise
        else:
            terminal.update(state='passed', finished=now())
            write_json(STUDY / 'completion.json', terminal)
            write_json(STUDY / 'status.json', terminal)
            log('All ADR qualification stages passed')
