#!/usr/bin/env python3
"""Finish the already authorized equilibrium work after the active driver."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

directory = output_directory('equilibrium_dt080_20261016') / 'background_finish'
root = directory.parent


def write(name, data):
    path = directory / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def run(script, *arguments):
    print('START', script, *arguments, flush=True)
    result = subprocess.run([sys.executable, str(CODE.parent / script), *arguments], cwd=root)
    print('END', script, result.returncode, flush=True)
    return result.returncode


lock = (directory / 'launcher.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
write('driver.json', {
    'pid': os.getpid(), 'pgid': os.getpgrp(), 'started': time.time(),
    'stop': f'kill -- -{os.getpgrp()}',
    'scope': 'Finish DOLFINx refinement, one prepared Ugly Pac-Man retry, saved-field checks and archive verification. No HDG runs.',
})
failures = []
try:
    previous_driver = json.loads((root / 'high_order_refinement.driver.json').read_text())
    print('WAIT for refinement driver', previous_driver['pid'], flush=True)
    with (root / 'high_order_refinement.lock').open('a') as previous_lock:
        fcntl.flock(previous_lock, fcntl.LOCK_EX)
        marker = root / 'high_order_refinement.completed.json'
        if not marker.exists():
            raise RuntimeError('Refinement driver exited without a completion marker; inspect its logs before resuming.')
        prior = json.loads(marker.read_text())
        if prior['finished'] < previous_driver['started']:
            raise RuntimeError('Refinement completion marker predates the active driver.')
        preserved = directory / 'refinement_before_retry.completed.json'
        if not preserved.exists():
            shutil.copy2(marker, preserved)

    if not (root / 'cg6_refined/uglypacman/completed.json').exists():
        if run('high_order_refinement_launcher.py', 'uglypacman'):
            failures.append('Prepared Ugly Pac-Man retry exited with an error.')
    accepted_refined = {
        geometry: (root / 'cg6_refined' / geometry / 'completed.json').exists()
        for geometry in ['star', 'uglypacman', 'iter', 'horseshoe']
    }
    for geometry, accepted in accepted_refined.items():
        if not accepted:
            failures.append(geometry + ': additional mesh refinement unavailable; retain accepted primary CG6 equilibrium.')

    if run('high_order_checks.py'):
        failures.append('Final saved-field checks exited with an error.')
    checks_marker = root / 'high_order_checks.completed.json'
    checks = json.loads(checks_marker.read_text()) if checks_marker.exists() else {}
    if checks.get('status') != 'complete':
        failures.extend(checks.get('failures') or ['Final checks did not complete.'])
    write('completed.json', {
        'status': 'complete' if not failures else 'partial',
        'finished': time.time(), 'failures': failures,
        'accepted_refined': accepted_refined,
        'results': str(root / 'RESULTS.md'),
        'gallery': str(root / 'final/density_bands.png'),
        'guiding_center_evolution_performed': False,
    })
    print('FINISHED', 'complete' if not failures else 'partial', flush=True)
except BaseException as error:
    write('failed.json', {'error': repr(error), 'finished': time.time()})
    traceback.print_exc()
    raise
