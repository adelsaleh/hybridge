"""Stand-in runner: works in steps with a child 'worker'; saves state and exits 2 on SIGUSR1."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from campaigns.common.lifecycle import REQUEUE_SIGNALS, CampaignStop, log, stop_on_signals, terminate_group
state = 'state.json'
done = json.load(open(state))['done'] if os.path.exists(state) else 0
log(f'runner pid {os.getpid()} restart {os.environ.get("SLURM_RESTART_COUNT", 0)} resuming at step {done}')
with stop_on_signals():
    try:
        while done < 1000:
            worker = subprocess.Popen(['sleep', '5'], start_new_session=True)
            worker.wait(); done += 1
            json.dump(dict(done=done), open(state, 'w'))
        sys.exit(0)
    except CampaignStop as stop:
        log(f'stop {stop.signal_name}: worker {terminate_group(worker)}, saved step {done}')
        json.dump(dict(done=done, stopped=stop.signal_name), open(state, 'w'))
        sys.exit(2 if stop.signal_name in REQUEUE_SIGNALS else 3)
