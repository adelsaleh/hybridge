"""Conservative shared-user GPU accounting immediately before submission."""
import getpass
import re
import subprocess


def count_requested_gpus(queue):
    """Count expanded running/pending array allocations; reject ambiguous GPU TRES."""
    count=0
    for line in queue.splitlines():
        job,name,state,resources,nodes=line.split('|')
        if resources in ('N/A','(null)','None',''): continue
        matches=re.findall(r'(?:^|,)(?:gres/)?gpu(?::[A-Za-z0-9_.-]+)?:(\d+)(?=,|$)',resources)
        if 'gpu' in resources and not matches: raise ValueError(f'Unrecognized GPU allocation: {line}')
        count+=sum(map(int,matches))*int(nodes)
    return count


def guard_gpu_submission(additional=1,limit=5):
    queue=subprocess.check_output(['squeue','--user='+getpass.getuser(),'--noheader','--array',
        '--format=%i|%j|%T|%b|%D'],text=True)
    existing=count_requested_gpus(queue)
    if existing+additional>limit:
        raise RuntimeError(f'Shared GPU ceiling: {existing} running/pending + {additional} requested > {limit}')
    return dict(existing_gpus=existing,additional_gpus=additional,limit=limit,queue=queue)
