#!/usr/bin/env python3
"""Wait for the active campaign, then retry horseshoe with smaller epsilon steps."""

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

directory = output_directory('equilibrium_dt080_eps0005_20261016') / 'horseshoe_recovery'
root = directory.parent


def write(name, data):
    path = directory / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


lock = (directory / 'launcher.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
write('driver.json', {'pid': os.getpid(), 'pgid': os.getpgrp(), 'started': time.time(),
                     'stop': f'kill -- -{os.getpgrp()}', 'scope': 'Serial horseshoe epsilon-continuation recovery after original campaign; no HDG runs'})
try:
    print('WAIT for original equilibrium campaign and quadrature checks', flush=True)
    with (root / 'launcher.lock').open('a') as previous:
        fcntl.flock(previous, fcntl.LOCK_EX)
    time.sleep(2)
    marker = root / 'completed.json'
    if marker.exists():
        shutil.copy2(marker, directory / 'original_campaign.completed.json')
        marker.unlink()
    print('START gentler horseshoe continuation', flush=True)
    result = subprocess.run([sys.executable, str(CODE.parent / 'launcher.py'),
                             '--geometries', 'horseshoe'], cwd=root)
    completion = json.loads(marker.read_text()) if marker.exists() else {}
    write('completed.json', {'exit_code': result.returncode, 'finished': time.time(),
                            'campaign_status': completion.get('status', 'incomplete'),
                            'failures': completion.get('failures', [])})
    (root / 'recovery_pending.json').unlink(missing_ok=True)
    print('FINISHED horseshoe recovery', completion.get('status'), flush=True)
except BaseException as error:
    write('failed.json', {'error': repr(error), 'finished': time.time()})
    (root / 'recovery_pending.json').unlink(missing_ok=True)
    traceback.print_exc()
    raise
