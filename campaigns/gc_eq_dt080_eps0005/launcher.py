#!/usr/bin/env python3
"""Detached, sequential epsilon continuation; accepted stages skip on resume."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

root = output_directory('equilibrium_dt080_eps0005_20261016')
repo = EQUILIBRIUM_REPO
archive = ARTICLE / 'equilibria'
python = FENICS_PYTHON
mpi = str(Path(python).parent / 'mpiexec')
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMBA_NUM_THREADS='1', MPLBACKEND='Agg')


def requested_ranks(geometry=None):
    control = root / 'rank_control.json'
    data = json.loads(control.read_text()) if control.exists() else {'mpi_ranks': 8}
    ranks = data.get('geometry_overrides', {}).get(geometry, data['mpi_ranks'])
    if type(ranks) is not int or not 1 <= ranks <= len(os.sched_getaffinity(0)):
        raise ValueError('Invalid MPI rank count: ' + str(ranks))
    return ranks


def write(name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def snapshot(label):
    sources = {}
    for directory in ['projects/diocotron/dolfinx', 'projects/diocotron/studies/equilibrium_catalog']:
        for source in (repo / directory).rglob('*.py'):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = root / 'producer_source' / digest / source.name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            sources[str(source.relative_to(repo))] = {
                'sha256': digest, 'snapshot': str(destination.relative_to(root))}
    write('execution/' + label + '.source.json', sources)


def execute(label, command):
    snapshot(label)
    log = root / 'logs' / (label + '_' + str(time.time_ns()) + '.log')
    log.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    print('START', label, str(log), flush=True)
    monitored = command[command.index('--output') + 1] if '--output' in command else None
    previous, last = {}, started
    peak_cpu = peak_rss = peak_ranks = 0
    ticks = os.sysconf('SC_CLK_TCK')
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=repo, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        write('active.json', {'label': label, 'pid': process.pid,
              'pgid': os.getpgrp(), 'command': command, 'log': str(log)})
        while process.poll() is None:
            time.sleep(3)
            now, current, rss = time.monotonic(), {}, 0
            if monitored:
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit():
                        continue
                    try:
                        raw = (p / 'cmdline').read_bytes()
                        args = raw.split(b'\0')
                        if monitored.encode() not in args or b'equilibrium_catalog.' not in raw:
                            continue
                        if b'mpiexec' in args[0]:
                            continue
                        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
                        current[p.name] = (int(fields[11]) + int(fields[12])) / ticks
                        rss += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                    except (OSError, ValueError):
                        continue
            increments = [max(0, value - previous[k]) for k, value in current.items() if k in previous]
            peak_cpu = max(peak_cpu, sum(increments) / (now - last))
            peak_ranks = max(peak_ranks, sum(value > .02 for value in increments))
            peak_rss = max(peak_rss, rss)
            previous, last = current, now
            info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(info['MemAvailable'].split()[0]) < 8 * 1024 * 1024:
                write('failed.json', {'error': 'Available memory below 8 GiB; stopped this campaign.',
                                      'stage': label, 'finished': time.time()})
                os.killpg(os.getpgrp(), signal.SIGTERM)
    record = dict(command=command, exit_code=process.returncode,
                  wall_seconds=time.monotonic() - started, peak_cpu_cores=peak_cpu,
                  peak_active_ranks=peak_ranks, peak_sum_rank_rss_bytes=peak_rss,
                  log=str(log))
    if command[0] == mpi:
        record['mpi_ranks'] = int(command[command.index('-n') + 1])
    write('execution/' + label + '.json', record)
    print('END', label, json.dumps(record), flush=True)
    return process.returncode


def solve(geometry, stage, parent):
    tag = stage['label']
    config = root / 'inputs' / geometry / (tag + '.json')
    output = root / 'stages' / geometry / tag
    completed = output / 'completed.json'
    if completed.exists():
        if json.loads((output / 'config.json').read_text()) != json.loads(config.read_text()):
            raise ValueError('Completed configuration differs: ' + str(output))
        print('SKIP accepted', geometry, tag, flush=True)
        return json.loads(completed.read_text())
    name = geometry + '_dt080_' + tag + '_cg6_20261016'
    preferred = requested_ranks(geometry)
    for ranks in ([preferred, 8] if preferred != 8 else [8]):
        if output.exists():
            output.rename(output.with_name(tag + '_attempt_' + str(time.time_ns())))
        command = [mpi, '-n', str(ranks), '-bind-to', 'core', python, '-u', '-m',
                   'projects.diocotron.studies.equilibrium_catalog.run_distance',
                   '--config', str(config), '--geometry', geometry, '--name', name,
                   '--output', str(output), '--archive', str(archive),
                   '--backend', 'mumps', '--target-distance', '.8',
                   '--distance-tolerance', '2e-7', '--maximum-points', '40',
                   '--warm-equilibrium', parent['archive_id']]
        if not execute(geometry + '_' + tag + '_ranks' + str(ranks), command):
            if ranks != preferred:
                control = json.loads((root / 'rank_control.json').read_text())
                control.setdefault('geometry_overrides', {})[geometry] = ranks
                write('rank_control.json', control)
            return json.loads(completed.read_text())
        failure = output / 'failed.json'
        reason = json.loads(failure.read_text()).get('error', '') if failure.exists() else ''
        if ranks == 8 or 'INVALID_RAY_ATLAS' not in reason:
            return None
        print('RANK_RETRY', geometry, tag, 'auxiliary flow atlas rejected; retry with accepted 8-rank partition', flush=True)
    return None


def select(geometry, stage, record, quadrature_check):
    link = root / 'accepted' / geometry
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink():
        link.unlink()
    if link.exists():
        raise ValueError('Refuse to replace a real directory: ' + str(link))
    link.symlink_to(Path('..') / 'stages' / geometry / stage['label'], target_is_directory=True)
    selected[geometry] = {**record, 'stage': stage['label'],
                          'quadrature_check': quadrature_check}
    write('selected.json', selected)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--geometries', nargs='+', choices=['star', 'uglypacman', 'iter', 'horseshoe'])
args = parser.parse_args()
lock = (root / 'launcher.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
write('driver.json', {'pid': os.getpid(), 'pgid': os.getpgrp(), 'started': time.time(),
                     'stop': f'kill -- -{os.getpgrp()}', 'mpi_ranks': requested_ranks(),
                     'scope': 'DOLFINx epsilon continuation only; no HDG runs'})
failures, targets = [], {}
selected = json.loads((root / 'selected.json').read_text()) if (root / 'selected.json').exists() else {}
try:
    schedule = json.loads((root / 'schedule.json').read_text())['stages']
    overrides_path = root / 'continuation_overrides.json'
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    seeds = json.loads((root / 'seeds.json').read_text())
    if args.geometries:
        seeds = {geometry: seeds[geometry] for geometry in args.geometries}
    # Complete the sharper bands on all four domains before the extra
    # quadrature solves. Every stage warm-starts from the preceding archive.
    for geometry, source in seeds.items():
        geometry_schedule = overrides.get(geometry, schedule)
        parent = source
        for stage in geometry_schedule[:-1]:
            parent = solve(geometry, stage, parent)
            if parent is None:
                failures.append(geometry + ': continuation failed at ' + stage['label'])
                break
        if parent is not None:
            targets[geometry] = parent
            select(geometry, geometry_schedule[-2], parent, 'pending')
            if execute('preview_' + geometry, [python, str(CODE / 'postprocess.py')]):
                failures.append(geometry + ': preview rendering failed')
    for geometry, parent in targets.items():
        geometry_schedule = overrides.get(geometry, schedule)
        result = solve(geometry, geometry_schedule[-1], parent)
        if result is None:
            failures.append(geometry + ': quadrature-64 solve failed; quadrature-48 field retained')
            select(geometry, geometry_schedule[-2], parent, 'solve failed')
            continue
        comparison = root / 'checks' / (geometry + '_quadrature.json')
        comparison.parent.mkdir(exist_ok=True)
        if not comparison.exists():
            command = [python, '-m', 'projects.diocotron.studies.equilibrium_catalog.check_refinement',
                       '--coarse', parent['checkpoint'], '--fine', result['checkpoint'],
                       '--quadrature', '16', '24', '--output', str(comparison)]
            if execute('compare_quadrature_' + geometry, command):
                failures.append(geometry + ': quadrature field comparison failed')
        select(geometry, geometry_schedule[-1], result,
               str(comparison.relative_to(root)) if comparison.exists() else 'comparison failed')
    if execute('final_plots', [python, str(CODE / 'postprocess.py')]):
        failures.append('Final rendering failed')
    for geometry in ['star', 'uglypacman', 'iter', 'horseshoe']:
        if geometry not in selected:
            failures.append(geometry + ': target equilibrium unavailable')
        elif not (root / 'checks' / (geometry + '_quadrature.json')).exists():
            failures.append(geometry + ': quadrature comparison unavailable')
    write('completed.json', {'status': 'complete' if not failures else 'partial',
          'failures': failures, 'finished': time.time(), 'target_relative_epsilon': .005,
          'accepted_geometries': sorted(selected), 'guiding_center_evolution_performed': False})
    print('FINISHED', 'complete' if not failures else 'partial', flush=True)
except BaseException as error:
    write('failed.json', {'error': repr(error), 'finished': time.time()})
    traceback.print_exc()
    raise
