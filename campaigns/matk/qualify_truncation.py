#!/usr/bin/env python3
"""Correctness and CUDA sanitizer qualification of native maximum-entry truncation.

Explicit build/executable paths keep historical comparisons reproducible after
build retirement. --plan never loads a matrix or initializes CUDA. Resume uses
identity-bound completion markers and verifies each completed log's digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from campaigns.common.lifecycle import terminate_group


REAL_MODES = ('dDDI', 'dDFI', 'dFFI')
BOUNDARY_CAPS = (8, 9, 16, 17, 32)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_plan():
    checks = []

    def add(label, build='fixed', mode='dDDI', test='oracle', cap=None,
            rows=None, fixture=None, sanitizer=None, expect_defect=False):
        checks.append(dict(label=label, build=build, mode=mode, test=test, cap=cap,
                           rows=rows, fixture=fixture, sanitizer=sanitizer,
                           expect_defect=expect_defect))

    for mode in REAL_MODES:
        for cap in range(1, 33):
            add(f'fixed-oracle-{mode}-cap{cap}', mode=mode, cap=cap)
        for cap in range(1, 9):
            add(f'baseline-control-{mode}-cap{cap}', build='baseline', mode=mode, cap=cap)
        for cap in BOUNDARY_CAPS:
            add(f'fixed-boundaries-{mode}-cap{cap}', mode=mode, test='boundaries', cap=cap)
    # A fresh CUDA context for each deliberately invalid baseline launch.
    for mode in ('dDDI', 'dFFI'):
        for cap in (9, 16, 17, 32):
            add(f'baseline-memcheck-{mode}-cap{cap}', build='baseline', mode=mode,
                cap=cap, fixture='unique', sanitizer='memcheck', expect_defect=True)
    for tool in ('memcheck', 'synccheck'):
        for mode in REAL_MODES:
            for cap in BOUNDARY_CAPS:
                add(f'fixed-{tool}-{mode}-cap{cap}', mode=mode, cap=cap, sanitizer=tool)
    # Each launch geometry takes a second grid-stride pass, including >524288 rows.
    for mode in REAL_MODES:
        for cap, rows in ((8, 524289), (16, 262145), (32, 131073)):
            add(f'fixed-gridstride-{mode}-cap{cap}', mode=mode, cap=cap, rows=rows,
                fixture='unique', sanitizer='memcheck')
    for mode in (*REAL_MODES, 'dZZI', 'dZCI', 'dCCI', 'hDDI', 'hDFI', 'hFFI'):
        add('fixed-rejections-' + mode, mode=mode, test='rejections')
    return checks


def evaluate_log(content, returncode, timed_out, *, library, sanitizer, expect_defect):
    summaries = [int(value) for value in re.findall(r'ERROR SUMMARY:\s+(\d+) errors?', content)]
    library_seen = f'AMGX library: {library}' in content
    oracle_pass = bool(re.search(r'^PASS$', content, re.MULTILINE))
    shared_fault = bool(re.search(r'Invalid __shared__ (?:read|write)', content))
    truncation_seen = 'truncate_kernel' in content
    hashes = {f'{kind}/rows{rows}/cap{cap}': value for kind, rows, cap, value in re.findall(
        r'fixture=(\w+) rows=(\d+) cap=(\d+).*output_fnv1a64=(\d+)', content)}
    if expect_defect:
        passed = (not timed_out and returncode not in (None, 0) and library_seen
                  and bool(summaries) and max(summaries) > 0 and shared_fault and truncation_seen)
    else:
        passed = not timed_out and returncode == 0 and oracle_pass and library_seen
        if sanitizer:
            passed = passed and bool(summaries) and max(summaries) == 0
    return dict(passed=bool(passed), sanitizer_errors=summaries, library_seen=library_seen,
                oracle_pass=oracle_pass, shared_memory_fault=shared_fault,
                truncation_kernel_seen=truncation_seen, oracle_hashes=hashes)


def main():
    repo = REPO
    builds = Path(os.environ.get('HYBRIDGE_BUILDS_ROOT', repo.parent / 'builds'))
    baseline_root = Path(os.environ.get('HYBRIDGE_AMGX_BASELINE_ROOT',
                                       repo / 'run_outputs/fork-integration-baselines/20261031'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=os.environ.get('HYBRIDGE_CAMPAIGN_OUT'))
    parser.add_argument('--fixed-build', type=Path, default=os.environ.get(
        'MATK_TRUNCATION_FIXED_BUILD', os.environ.get('HYBRIDGE_AMGX_BUILD_ROOT',
                                                     builds / 'AMGX/cuda13.1')))
    parser.add_argument('--baseline-build', type=Path, default=os.environ.get(
        'MATK_TRUNCATION_BASELINE_BUILD', baseline_root / 'main'))
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--sanitizer-timeout', type=float, default=600)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    fixed, baseline = args.fixed_build.resolve(), args.baseline_build.resolve()
    executable = (args.executable or fixed / 'check_truncation').resolve()
    checks = check_plan()
    if args.plan:
        print(json.dumps(dict(fixed_build=str(fixed), baseline_build=str(baseline),
                              executable=str(executable), check_count=len(checks), checks=checks), indent=2))
        return 0
    if args.output_dir is None:
        parser.error('--output-dir or HYBRIDGE_CAMPAIGN_OUT is required')
    if args.timeout <= 0 or args.sanitizer_timeout <= 0:
        parser.error('timeouts must be positive')
    for path in (fixed / 'libamgxsh.so', baseline / 'libamgxsh.so', executable):
        if not path.is_file():
            parser.error(f'missing {path}; build the explicit qualification inputs first')
    if fixed == baseline or sha256(fixed / 'libamgxsh.so') == sha256(baseline / 'libamgxsh.so'):
        parser.error('fixed and baseline libraries must differ; baseline reproduction must be independent')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(fixed_build=str(fixed), baseline_build=str(baseline),
                    executable=str(executable), fixed_library_sha256=sha256(fixed / 'libamgxsh.so'),
                    baseline_library_sha256=sha256(baseline / 'libamgxsh.so'),
                    harness_sha256=sha256(executable), runner_sha256=sha256(__file__),
                    timeout=args.timeout, sanitizer_timeout=args.sanitizer_timeout, checks=checks)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity_file = out / 'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        parser.error('qualification inputs changed; choose a new output directory')
    identity_file.write_text(json.dumps(identity, indent=2) + '\n')
    sys.path.insert(0, str(repo))
    from campaigns.common.protocol import host_provenance
    host_provenance(out, qualification='native-truncation', identity=identity,
                    gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,uuid',
                                                 '--format=csv,noheader'], text=True).strip())
    target = out / 'qualification.json'
    report = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'), identity=identity,
                  identity_sha256=identity_sha, checks=[])
    if target.exists():
        old = json.loads(target.read_text())
        if old.get('identity_sha256') != identity_sha:
            parser.error('stored report does not match qualification identity')
    report.update({key: value for key, value in identity.items() if key.endswith('_sha256')})

    def save():
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(target)

    def run(spec):
        label = spec['label']
        log, done = out / (label + '.log'), out / (label + '.done')
        if done.exists() and log.exists():
            row = json.loads(done.read_text())
            if (row.get('identity_sha256') == identity_sha and row.get('passed')
                    and row.get('log_sha256') == sha256(log)):
                report['checks'].append(row)
                print('[matK truncation] Already passed ' + label, flush=True)
                save()
                return
        build = fixed if spec['build'] == 'fixed' else baseline
        env = os.environ.copy()
        env['HYBRIDGE_AMGX_BUILD_ROOT'] = str(build)
        for suffix, key in (('CAP', 'cap'), ('ROWS', 'rows'), ('CASE', 'fixture')):
            name = 'AMGX_TRUNCATION_TEST_' + suffix
            if spec[key] is None:
                env.pop(name, None)
            else:
                env[name] = str(spec[key])
        command = ['bash', str(repo / 'scripts/gpu/run_cuda13.sh')]
        if spec['sanitizer']:
            cuda = Path(env.get('HYBRIDGE_CUDA13_ROOT', '/usr/local/cuda-13.1'))
            command += [str(cuda / 'bin/compute-sanitizer'), '--tool', spec['sanitizer'],
                        '--error-exitcode', '99', '--print-limit', '20']
        command += [str(executable), spec['mode'], spec['test']]
        print('[matK truncation] Starting ' + label, flush=True)
        started, timed_out = time.perf_counter(), False
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.sanitizer_timeout if spec['sanitizer'] else args.timeout)
            except subprocess.TimeoutExpired:
                terminate_group(process, grace=5.0)
                returncode, timed_out = process.returncode, True
            except BaseException:
                terminate_group(process, grace=5.0)
                raise
        content = log.read_text(errors='replace')
        row = dict(spec, build=str(build), returncode=returncode, timed_out=timed_out,
                   identity_sha256=identity_sha, command=command,
                   elapsed_seconds=time.perf_counter() - started, log=str(log), log_sha256=sha256(log))
        row.update(evaluate_log(content, returncode, timed_out, library=build / 'libamgxsh.so',
                                sanitizer=spec['sanitizer'], expect_defect=spec['expect_defect']))
        report['checks'].append(row)
        if row['passed']:
            done.write_text(json.dumps(row, indent=2) + '\n')
        save()
        print(f"[matK truncation] {label}: {'PASS' if row['passed'] else 'FAIL'} "
              f"({row['elapsed_seconds']:.3f} s, exit={returncode}, errors={row['sanitizer_errors']})", flush=True)

    save()
    for spec in checks:
        run(spec)
    indexed = {row['label']: row for row in report['checks']}
    parity = {}
    for mode in REAL_MODES:
        for cap in range(1, 9):
            actual = indexed[f'fixed-oracle-{mode}-cap{cap}']['oracle_hashes']
            reference = indexed[f'baseline-control-{mode}-cap{cap}']['oracle_hashes']
            parity[f'{mode}/cap{cap}'] = bool(actual) and actual == reference
    report['unchanged_safe_caps_bitwise_parity'] = parity
    report['status'] = ('passed' if all(row['passed'] for row in report['checks'])
                        and all(parity.values()) else 'failed')
    save()
    if report['status'] == 'passed':
        (out / 'COMPLETE').write_text(identity_sha + '\n')
    print('[matK truncation] Qualification ' + report['status'], flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
