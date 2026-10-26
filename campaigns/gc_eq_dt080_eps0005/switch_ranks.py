#!/usr/bin/env python3
"""Let the active worker finish, then resume the detached driver with more ranks."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

root = output_directory('equilibrium_dt080_eps0005_20261016')
lock = (root / 'rank_switch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
old = json.loads((root / 'driver.json').read_text())
old_pid = old['pid']
worker_pid = None
paused = False


def write(name, data):
    path = root / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def state(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return None


def terminate_old():
    global paused
    if state(old_pid) not in [None, 'Z']:
        os.kill(old_pid, signal.SIGTERM)
        if paused:
            os.kill(old_pid, signal.SIGCONT)
    paused = False


def stop(signum, frame):
    terminate_old()
    if worker_pid is not None and state(worker_pid) not in [None, 'Z']:
        try:
            os.kill(worker_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
write('rank_switch.driver.json', {
    'pid': os.getpid(), 'pgid': os.getpgrp(), 'started': time.time(),
    'previous_driver': old, 'next_mpi_ranks': 12,
    'stop': f'kill -- -{os.getpgrp()} -{old["pgid"]}',
})
try:
    old_process = Path('/proc') / str(old_pid)
    if old_process.exists():
        if (old_process / 'cwd').resolve() != root:
            raise RuntimeError('Previous driver has a different working directory')
        command = (old_process / 'cmdline').read_bytes().split(b'\0')
        if b'launcher.py' not in command:
            raise RuntimeError('Previous process is not the campaign launcher')
        # Pause only the coordinating Python process. Its existing MPI solve
        # continues; pausing prevents a new stage from starting at 8 ranks.
        os.kill(old_pid, signal.SIGSTOP)
        paused = True
        while state(old_pid) not in ['T', 't', None, 'Z']:
            time.sleep(.05)
    active = json.loads((root / 'active.json').read_text())
    worker_pid = active['pid']
    print('WAIT for existing worker', worker_pid, active['label'], flush=True)
    write('rank_switch.waiting.json', active)
    while state(worker_pid) not in [None, 'Z']:
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(info['MemAvailable'].split()[0]) < 8 * 1024 * 1024:
            stop(signal.SIGTERM, None)
        time.sleep(2)
    # Preserve the old driver's records. A paused coordinator cannot collect
    # its child's final CPU samples; report this rather than inventing values.
    (root / 'rank_switch_previous_driver.json').write_text(json.dumps(old, indent=2) + '\n')
    write('rank_switch_previous_worker.json', {
        **active, 'observed_finished': time.time(),
        'note': 'Coordinator paused for rank switch; use the case manifest for solver duration. No final peak-CPU estimate recovered.',
    })
    terminate_old()
    with (root / 'launcher.lock').open('a') as previous_lock:
        fcntl.flock(previous_lock, fcntl.LOCK_EX)
    print('RESUME accepted stages with configured MPI ranks', flush=True)
    with (root / 'driver_12ranks_stdout.log').open('a') as stream:
        child = subprocess.Popen([sys.executable, str(CODE / 'launcher.py')],
                                 cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        write('rank_switch.resumed.json', {'pid': child.pid, 'pgid': os.getpgrp(),
                                          'started': time.time(), 'mpi_ranks': 12})
        result = child.wait()
    marker = root / 'completed.json'
    completion = json.loads(marker.read_text()) if marker.exists() else {}
    write('rank_switch.completed.json', {'exit_code': result, 'finished': time.time(),
          'campaign_status': completion.get('status', 'incomplete')})
except BaseException as error:
    # Never leave the old coordinator stopped if preparation itself fails.
    if paused:
        try:
            os.kill(old_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    write('rank_switch.failed.json', {'error': repr(error), 'finished': time.time()})
    traceback.print_exc()
    raise
