"""Bounded, resumable Q1 batches. Requires a passing smoke marker and a GPU job.

Use the plan module for inspection. Each task has isolated attempts and source/
specification hashes; failures remain in the report and never become successes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from campaigns.common.lifecycle import (REQUEUE_SIGNALS, CampaignStop, describe_release, device_state, log,
                                        release_device, stop_on_signals, stop_signal_name, tail, terminate_group)
from campaigns.common.production_placement import require_placement
from campaigns.common.timing_summary import format_rows, summarize
from campaigns.h100_assembly.plan import ROOT, build_tasks

# Attempts that never finished; resumed batches start them again.
RETRYABLE = {'running', 'interrupted', 'budget_exhausted'}
# Finished without a measurement for a reason of the task, not of the run: an unsupported
# configuration, or one allocation larger than the whole device (memory_excluded).
EXCLUDED = {'unsupported', 'memory_excluded'}
OOM_REQUEST = re.compile(r'out of memory allocating ([\d,]+) bytes')


def oversized_request(log_text: str, device: dict) -> tuple[int, int] | None:
    """(largest failed request, device total) in bytes when a failed request exceeds the device."""
    requests = [int(value.replace(',', '')) for value in OOM_REQUEST.findall(log_text)]
    totals = [int(entry['total_mib']) * 2**20 for entry in (device or {}).get('memory', []) if 'total_mib' in entry]
    if requests and totals and max(requests) > max(totals):
        return max(requests), max(totals)
    return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu_state():
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader,nounits']
    devices = os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    if devices:
        command += ['--id', devices]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        if error.returncode != 6 or '--id' not in command:
            raise
        # Slurm's physical index is absent from the job's device namespace, which shows only the
        # allocated GPUs (as hybridge.runtime.benchmarking.gpu_compute_processes handles).
        result = subprocess.run(command[:-2], text=True, capture_output=True, check=True)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1:
        raise RuntimeError('full Q1 batch must identify exactly one allocated GPU')
    name, uuid, driver, memory = rows[0]
    return dict(name=name, uuid=uuid, driver=driver, memory_mib=int(memory))


def record_timings(task, row, output, artifact):
    """Log a task's measured timings now and append them to timings.jsonl for live evaluation."""
    try:
        rows = summarize(artifact)
    except (OSError, ValueError) as error:
        log(f'timings {task.id}: summary unavailable ({type(error).__name__}: {error})')
        return
    for line in format_rows(rows) or ['  no timing fields found in the artifact']:
        print(f'timings {task.id}{line}', flush=True)
    with (output / 'timings.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(), job=os.environ.get('SLURM_JOB_ID'), task=task.id,
                                     group=task.group, order=task.order, precision=task.precision,
                                     wall_seconds=row['wall_seconds'], artifact=str(artifact),
                                     rows=[dict(label=label, ms=metrics) for label, metrics in rows])) + '\n')


def heartbeat_memory():
    try:
        memory = device_state()['memory']
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return 'device memory unavailable'
    return ', '.join(f'gpu{row["index"]} {row["used_mib"]}/{row["total_mib"]} MiB' for row in memory)


def validate_artifact(task, path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('missing output artifact')
    values = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
              if path.suffix == '.jsonl' else [json.loads(path.read_text())])
    if not values or any(not isinstance(value, dict) or not value for value in values):
        raise ValueError('empty or invalid benchmark output')
    if any(value.get('complete') is False for value in values):
        raise ValueError('benchmark reports incomplete output')
    if task.group == 'profiles' and (not values[0].get('runs') or any(row.get('returncode') != 0 for row in values[0]['runs'])):
        raise ValueError('profiler reports failed runs')
    root = path
    for _ in Path(task.artifact).parts:
        root = root.parent
    for extra in task.additional_artifacts:
        if not (root / extra).is_file() or not (root / extra).stat().st_size:
            raise ValueError(f'missing profiler artifact: {extra}')
    return digest(path)


def run_task(task, output, source_identity, *, retry_failed=False, timeout=None):
    directory = output / 'tasks' / task.id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read_json(directory / 'result.json')
        identity = task.specification_sha256()
        if prior:
            if prior.get('specification_sha256') != identity or prior.get('source_identity') != source_identity:
                raise ValueError(f'{task.id}: changed sources/specification; use a new output root')
            if prior.get('status') in {'passed', 'unsupported'}:
                if validate_artifact(task, Path(prior['artifact'])) != prior.get('artifact_sha256'):
                    raise ValueError(f'{task.id}: completed artifact changed')
                if any(digest(Path(path)) != expected for path, expected in prior.get('additional_artifact_sha256', {}).items()):
                    raise ValueError(f'{task.id}: completed profiler artifact changed')
                log(f'reuse {task.id}: {prior["status"]} (attempt {prior.get("attempt")}, artifact verified)')
                return prior
            if prior.get('status') not in RETRYABLE and not retry_failed:
                log(f'keep {task.id}: {prior.get("status")} (attempt {prior.get("attempt")}); pass --retry-failed to rerun')
                return prior
            log(f'resume {task.id}: previous attempt {prior.get("attempt")} ended {prior.get("status")}')
        attempt = 1
        while (directory / f'attempt_{attempt}').exists():
            attempt += 1
        work = directory / f'attempt_{attempt}'
        work.mkdir()
        artifact = work / task.artifact
        artifact.parent.mkdir(parents=True, exist_ok=True)
        command = task.command(work)
        row = dict(id=task.id, group=task.group, order=task.order, attempt=attempt, status='running',
                   specification_sha256=identity, source_identity=source_identity, artifact=str(artifact),
                   command=command, precision=task.precision, mode=task.mode, stages=list(task.stages))
        write_json(work / 'specification.json', row)
        write_json(directory / 'result.json', row)
        if task.unsupported_reason:
            write_json(artifact, dict(status='unsupported', reason=task.unsupported_reason, measured=False))
            row.update(status='unsupported', reason=task.unsupported_reason,
                       artifact_sha256=digest(artifact), wall_seconds=0.)
            write_json(work / 'result.json', row)
            write_json(directory / 'result.json', row)
            return row
        started = time.monotonic()
        stopped = None
        limit = min(task.timeout_seconds, timeout) if timeout is not None else task.timeout_seconds
        log(f'start {task.id}: attempt {attempt}, {task.precision}, timeout {limit:.0f}s, log {work / "worker.log"}')
        with (work / 'worker.log').open('w') as worker_log:
            stdout = artifact.open('w') if task.stdout_artifact else worker_log
            process = None
            try:
                environment = dict(os.environ, HYBRIDGE_PRECISION=task.precision, PYTHONDONTWRITEBYTECODE='1')
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout,
                                           stderr=worker_log, start_new_session=True)
                row['pid'] = process.pid
                write_json(directory / 'result.json', row)
                while True:
                    remaining = limit - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, limit)
                    try:
                        code = process.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        log(f'running {task.id}: {time.monotonic()-started:.0f}s of {limit:.0f}s; {heartbeat_memory()}; '
                            f'last output: {tail(work / "worker.log", 1)[-200:] or "none yet"}')
                if task.stdout_artifact:
                    stdout.flush()
                row.update(returncode=code, status='failed' if code else 'passed')
                if stop_signal_name(code):
                    row.update(status='interrupted', stop_signal=stop_signal_name(code))
                if code == 0:
                    row['artifact_sha256'] = validate_artifact(task, artifact)
                    row['additional_artifact_sha256'] = {str(work / extra): digest(work / extra)
                                                         for extra in task.additional_artifacts}
            except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
                if isinstance(error, KeyboardInterrupt):
                    stopped = getattr(error, 'signal_name', 'SIGINT')
                budget_timeout = not stopped and timeout is not None and timeout < task.timeout_seconds
                row.update(status='interrupted' if stopped else 'budget_exhausted' if budget_timeout else 'timeout',
                           error=type(error).__name__)
                if stopped:
                    row['stop_signal'] = stopped
            except (OSError, ValueError) as error:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
            finally:
                if process:
                    row['worker_exit'] = terminate_group(process)
                if task.stdout_artifact:
                    stdout.close()
        row['wall_seconds'] = time.monotonic() - started
        row['device'] = release_device()
        log_text = (work / 'worker.log').read_text(errors='replace').lower()
        if row['status'] == 'failed':
            row['failure_kind'] = ('out_of_memory' if 'out of memory' in log_text or 'outofmemoryerror' in log_text else
                                  'resource_limit' if 'shared memory' in log_text else
                                  'unsupported' if 'notimplementederror' in log_text else 'benchmark_failure')
            oversized = oversized_request(log_text, row['device']) if row['failure_kind'] == 'out_of_memory' else None
            if oversized:
                row.update(status='memory_excluded', requested_bytes=oversized[0], device_total_bytes=oversized[1])
        write_json(work / 'result.json', row)
        write_json(directory / 'result.json', row)
        log(f'finish {task.id}: {row["status"]}' + (f' ({row["failure_kind"]})' if 'failure_kind' in row else '')
            + f', returncode {row.get("returncode")}, {row["wall_seconds"]:.1f}s, {describe_release(row["device"])}')
        if row['status'] not in {'passed', 'interrupted'} | EXCLUDED:
            print(tail(work / 'worker.log'), flush=True)
        if row['status'] == 'passed':
            record_timings(task, row, output, artifact)
        if stopped:
            raise CampaignStop(stopped)
        return row


def gpu_counters_restricted(params=Path('/proc/driver/nvidia/params')):
    """Nsight needs counters the driver reserves for root when RmProfilingAdminOnly is 1."""
    try:
        text = params.read_text()
    except OSError:
        return False
    return os.geteuid() != 0 and any(line.split() == ['RmProfilingAdminOnly:', '1'] for line in text.splitlines())


def passing_smoke(path):
    if path.is_dir():
        attempts = list(path.glob('attempt_*/completion.json'))
        if not attempts:
            raise ValueError('no smoke completion marker')
        path = max(attempts, key=lambda marker: int(marker.parent.name.split('_')[-1]))
    if read_json(path).get('status') != 'passed':
        raise ValueError('latest smoke attempt has not passed')
    return path


def pilot_tasks():
    identifiers = {
        f'{equation}-recovery-p{p}-n16-legendre-modal-{coefficient}-{cache}-{flux}'
        for p in (1, 9)
        for equation, coefficient, cache, flux in (
            ('poisson', 'scalar', 'schur-lu', 'l2_closest'),
            ('adr', 'variable-full', 'schur-lu+mass', 'l2_closest'),
            ('adr', 'variable-full', 'schur-lu+mass', 'RT_projection'))
    }
    identifiers |= {'ar-partial-p1-n16-legacy-lagrange-coo-b32',
                    'ar-partial-p6-n16-legendre-modal-coo-b32', 'tensor-primal-p9-n16',
                    'ar-p9-h0.02-legendre-modal-float64-cutensor-magma',
                    'ar-layout-p9-h0.02-legendre-modal-b32'}
    selected = [task for task in build_tasks() if task.id in identifiers]
    if {task.id for task in selected} != identifiers:
        raise ValueError('expanded pilot registry is incomplete')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-completion', type=Path, required=True)
    parser.add_argument('--group')
    parser.add_argument('--exclude-profiles', action='store_true', help='Defer profiler tasks when cluster counters are restricted')
    parser.add_argument('--order', type=int, choices=range(1, 10))
    parser.add_argument('--shard', default='0/1')
    parser.add_argument('--budget-minutes', type=float, default=120)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    require_placement('device')
    try:
        args.smoke_completion = passing_smoke(args.smoke_completion)
    except ValueError as error:
        parser.error(str(error))
    if args.pilot and (args.group or args.order is not None):
        parser.error('pilot must run its complete predefined selection')
    if not args.exclude_profiles and gpu_counters_restricted():
        print('GPU performance counters are restricted to administrators; deferring profiler tasks.', flush=True)
        args.exclude_profiles = True
    try:
        index, count = map(int, args.shard.split('/'))
        if not 0 <= index < count or not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
            raise ValueError
    except ValueError:
        parser.error('require shard i/N with 0 <= i < N and positive budget')
    registry = pilot_tasks() if args.pilot else build_tasks()
    if args.group and set(args.group.split(',')) - {task.group for task in registry}:
        parser.error('unknown task group')
    tasks = [task for task in registry if (not args.group or task.group in args.group.split(','))
             and (not args.exclude_profiles or not task.mode.startswith('profile'))
             and (args.order is None or task.order == args.order)
             and int(hashlib.sha256(task.id.encode()).hexdigest(), 16) % count == index]
    if not tasks:
        parser.error('selection contains no tasks')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'shard_{index}_of_{count}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from scripts.reports.run_provenance import source_snapshot, changed_snapshot_files, run_provenance
        source_paths = [path.relative_to(ROOT) for folder in ('hybridge', 'scripts', 'campaigns', 'configs', 'vendor')
                        for path in (ROOT / folder).rglob('*') if path.is_file() and path.suffix in ('.py', '.cu', '.cuh', '.json', '.sh', '.sbatch')]
        job = output / 'batches' / f'{os.environ["SLURM_JOB_ID"]}_{index}_{time.time_ns()}'
        snapshot = source_snapshot(ROOT, source_paths, job / 'source.tar.gz')
        libraries = {}
        for name, path in (
            ('amgx', Path(os.environ['HYBRIDGE_AMGX_BUILD_ROOT']) / 'libamgxsh.so'),
            ('magma', Path(os.environ['HYBRIDGE_MAGMA_LIBRARY'])),
        ):
            libraries[name] = dict(path=str(path.resolve()), sha256=digest(path))
        provenance = run_provenance(ROOT)
        device = gpu_state()
        identity = hashlib.sha256(json.dumps(dict(sources=snapshot['files_sha256'], libraries=libraries,
            packages=provenance['packages'], device={k:v for k,v in device.items() if k != 'uuid'},
            cuda_runtime=provenance.get('cuda_runtime')), sort_keys=True).encode()).hexdigest()
        write_json(job / 'provenance.json', dict(provenance, libraries=libraries, device=device, snapshot=snapshot, source_identity=identity,
            smoke_path=str(args.smoke_completion.resolve()), smoke_sha256=digest(args.smoke_completion),
            affinity=sorted(os.sched_getaffinity(0)), selection=[task.id for task in tasks]))
        prior = Counter(read_json(output / 'tasks' / task.id / 'result.json').get('status', 'new') for task in tasks)
        log(f'shard {index}/{count} job {os.environ["SLURM_JOB_ID"]}: {len(tasks)} tasks, budget {args.budget_minutes:g} min, '
            f'prior states {dict(prior)}; batch records {job}')
        baseline = release_device()
        log(f'before first task: {describe_release(baseline)}')
        stop = time.monotonic() + args.budget_minutes * 60
        rows, stop_reason, current = [], None, None
        state_path = output / f'shard_{index}_of_{count}.state.json'

        def save_state():
            # Resume point after any exit: what ran, what is left, and why it stopped.
            done = {row['id'] for row in rows}
            write_json(state_path, dict(job=os.environ['SLURM_JOB_ID'], batch=str(job), updated=time.time(),
                                        current=current, stop_reason=stop_reason, completed=len(rows), selected=len(tasks),
                                        results=dict(Counter(row['status'] for row in rows)),
                                        remaining=[task.id for task in tasks if task.id not in done]))

        with stop_on_signals():
            try:
                if baseline.get('released') is False:
                    stop_reason = 'device_not_released'
                for position, task in enumerate(tasks if stop_reason is None else (), 1):
                    if time.monotonic() >= stop:
                        stop_reason = 'budget'
                        break
                    current = task.id
                    save_state()
                    log(f'task {position}/{len(tasks)}: {task.id} ({task.group}, p={task.order})')
                    rows.append(run_task(task, output, identity, retry_failed=args.retry_failed,
                                         timeout=stop-time.monotonic()))
                    current = None
                    write_json(job / 'progress.json', dict(tasks=rows, total=len(tasks)))
                    save_state()
                    if rows[-1]['status'] == 'budget_exhausted':
                        stop_reason = 'budget'
                        break
                    if rows[-1].get('device', {}).get('released') is False:
                        stop_reason = 'device_not_released'
                        break
            except CampaignStop as error:
                stop_reason = error.signal_name
            finally:
                # Signal handlers stay installed here, so a repeated SIGTERM cannot cut the save short.
                changes = changed_snapshot_files(ROOT, snapshot)
                incomplete = len(rows) < len(tasks) or any(row['status'] == 'budget_exhausted' for row in rows)
                status = ('source_changed' if changes else 'device_not_released' if stop_reason == 'device_not_released' else
                          'interrupted' if stop_reason and stop_reason.startswith('SIG') else
                          'batch_incomplete' if incomplete else
                          'completed_with_failures' if any(row['status'] not in {'passed'} | EXCLUDED for row in rows) else
                          'passed_with_unsupported' if any(row['status'] in EXCLUDED for row in rows) else 'passed')
                summary = dict(status=status, stop_reason=stop_reason, attempted=len(rows), selected=len(tasks),
                               results=dict(Counter(row['status'] for row in rows)), source_changes=changes, tasks=rows)
                write_json(job / 'completion.json', summary)
                write_json(output / f'shard_{index}_of_{count}.completion.json', summary)
                save_state()
                log(json.dumps({k: v for k, v in summary.items() if k != 'tasks'}))
                if status not in {'passed', 'passed_with_unsupported'}:
                    unfinished = len(tasks) - sum(row['status'] in {'passed'} | EXCLUDED for row in rows)
                    log(f'resume state {state_path}; rerun the same command to continue ({unfinished} tasks not passed)')
        # 2 asks job.sbatch to requeue the same shard; 3 is a deliberate stop that must not requeue.
        if status == 'batch_incomplete' or stop_reason in REQUEUE_SIGNALS:
            return 2
        return 0 if status in {'passed', 'passed_with_unsupported'} else 3 if status == 'interrupted' else 1


if __name__ == '__main__':
    sys.exit(main())
