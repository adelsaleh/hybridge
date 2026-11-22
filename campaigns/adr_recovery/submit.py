"""Prepare or submit the bounded recovery campaign with a unique Slurm name."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

from campaigns.adr_recovery.run import REPO, freeze, plan
from campaigns.common.slurm_gpu import guard_gpu_submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    label='smoke' if args.smoke else 'large'
    out=(args.output or REPO/'run_outputs'/f'adr-recovery-{label}-{stamp}').resolve()
    values=dict(seconds=600,repeats=2,min_nx=8,max_nx=8,chunk_limit=512) if args.smoke else dict(
        seconds=8700,repeats=3,min_nx=192,max_nx=512,chunk_limit=8192)
    config=plan(argparse.Namespace(**values))
    name=f'adr-recovery-p2-p9-{label}-reuse-rebuild-{stamp}'
    command=['sbatch','--parsable','--job-name='+name,'--output='+str(out/'slurm-%j.log'),
        '--export=ALL,HYBRIDGE_REPO='+str(REPO)+',HYBRIDGE_PYTHON='+sys.executable+',HYBRIDGE_CAMPAIGN_OUT='+str(out)]
    if args.smoke: command+=['--time=00:15:00']
    command += [str(REPO/'campaigns/adr_recovery/run.sbatch')]
    for key,value in values.items(): command += ['--'+key.replace('_','-'),str(value)]
    record=dict(job_name=name,output=str(out),command=command,configuration=config,status='planned')
    if args.execute:
        if (out/'launch.json').exists(): raise RuntimeError('Output already has a launch receipt; use a new output or explicit manual resume')
        manifest=freeze(out,config)
        record['source_identity']=manifest['sha256']
        record['allocation_guard']=guard_gpu_submission()
        job=subprocess.check_output(command,cwd=REPO,text=True).strip().split(';')[0]
        record.update(job_id=job,status='submitted')
        (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
        records=REPO/'docs/campaigns/adr_recovery/records';records.mkdir(parents=True,exist_ok=True)
        (records/f'launch-{job}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
