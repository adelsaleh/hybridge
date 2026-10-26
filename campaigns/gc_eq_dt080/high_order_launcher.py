#!/usr/bin/env python3
"""Detached, sequential and resumable equilibrium preparation."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, traceback

root=output_directory('equilibrium_dt080_20261016')
repo=EQUILIBRIUM_REPO
archive=ARTICLE / 'equilibria'
python=FENICS_PYTHON
mpi=str(Path(python).parent/'mpiexec')
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLBACKEND='Agg')
phase='high_order'

def write(name,data):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def wait_for_case(output):
    # A diagnostic retry may already own this case when the sequential
    # campaign reaches it. Wait before inspecting or renaming its directory.
    while True:
        active=False
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                args=(p/'cmdline').read_bytes().split(b'\0')
                if (str(output).encode() in args
                        and b'projects.diocotron.studies.equilibrium_catalog.run_distance' in args):
                    active=True;break
            except OSError:pass
        if not active:return
        time.sleep(3)

def execute(label,command):
    started=time.monotonic();peak_cores=0.;peak_rss=0;active_ranks=0
    previous={};last=started;ticks=os.sysconf('SC_CLK_TCK')
    logfile=root/f'{label}_{int(time.time())}.log'
    monitored_argument=command[command.index('--output')+1] if '--output' in command else command[-1]
    sources={}
    for directory in ['projects/diocotron/dolfinx', 'projects/diocotron/studies/equilibrium_catalog']:
        for source in (repo/directory).rglob('*.py'):
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            destination=root/'producer_source'/digest/source.name
            if not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
            sources[str(source.relative_to(repo))]=dict(sha256=digest,snapshot=str(destination.relative_to(root)))
    write(f'{label}.source.json',sources)
    print('START '+label+' '+str(logfile),flush=True)
    worker_env=env.copy()
    if 'projects.diocotron.studies.equilibrium_catalog.check_distance' in command:
        worker_env['NUMBA_NUM_THREADS']='16'
    with logfile.open('w') as stream:
        process=subprocess.Popen(command,cwd=repo,env=worker_env,stdout=stream,stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(2);now=time.monotonic();current={};rss=0
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    cmd=(p/'cmdline').read_bytes();args=cmd.split(b'\0')
                    if monitored_argument.encode() not in args or b'equilibrium_catalog.' not in cmd or b'mpiexec' in args[0]:continue
                    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                    current[p.name]=(int(fields[11])+int(fields[12]))/ticks
                    rss+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
                except (OSError,ValueError):pass
            increments=[max(0,v-previous[k]) for k,v in current.items() if k in previous]
            peak_cores=max(peak_cores,sum(increments)/(now-last))
            active_ranks=max(active_ranks,sum(v>.02 for v in increments));peak_rss=max(peak_rss,rss)
            previous=current;last=now
            meminfo=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(meminfo['MemAvailable'].split()[0]) < 8*1024*1024:
                print('MEMORY_RESERVE_STOP '+label,flush=True)
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
                break
    record=dict(command=command,exit_code=process.returncode,wall_seconds=time.monotonic()-started,
                peak_cpu_cores=peak_cores,peak_active_ranks=active_ranks,peak_sum_rank_rss_bytes=peak_rss,
                monitored_argument=monitored_argument,log=str(logfile))
    write(f'{label}.execution.json',record)
    print('END '+label+' '+json.dumps(record),flush=True)
    return process.returncode

import fcntl
lock = (root/'high_order.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
write('high_order.driver.json', dict(pid=os.getpid(), pgid=os.getpgid(0),
    stop=f'kill -- -{os.getpgid(0)}', started=time.time(),
    scope='DOLFINx equilibrium computation only; no HDG preservation tests',
    mpi_ranks=8, maximum_degree=6))
failures=[]
try:
    for geometry in ['star', 'uglypacman', 'iter', 'horseshoe']:
        for degree in [4, 6]:
            level=f'cg{degree}'
            output=root/level/geometry
            if (output/'completed.json').exists():
                print('SKIP '+level+'/'+geometry, flush=True)
                continue
            parent=root/('cg4' if degree==6 else 'resolved')/geometry
            if degree==4 and not (parent/'completed.json').exists():
                parent=root/'placement'/geometry
            if not (parent/'completed.json').exists():
                failures.append(level+'/'+geometry+': source not available')
                continue
            source=json.loads((parent/'completed.json').read_text())['archive_id']
            config=json.loads((root/'placement'/geometry/'config.json').read_text())
            config.update(degree=degree, quadrature_degree=32, backend='mpi', threads=1)
            if geometry=='iter':config['auxiliary_quadrature_degree']=24
            inputs=root/'inputs'/level/geometry
            inputs.mkdir(parents=True,exist_ok=True)
            config_path=inputs/'config.json'
            if config_path.exists() and json.loads(config_path.read_text())!=config:
                raise ValueError('Changed configuration for '+str(config_path))
            write(str(config_path.relative_to(root)), config)
            wait_for_case(output)
            resume=(output/'branch'/'run.json').exists()
            if output.exists() and not resume:
                output.rename(output.with_name(geometry+'_attempt_'+str(int(time.time()))))
            command=[mpi,'-n','8','-bind-to','core',python,'-u','-m',
                'projects.diocotron.studies.equilibrium_catalog.run_distance',
                '--config',str(config_path),'--geometry',geometry,
                '--name',geometry+'_dt080_'+level+'_20261016',
                '--output',str(output),'--archive',str(archive),
                '--target-distance','.8','--distance-tolerance','2e-7']
            command+=['--resume'] if resume else ['--warm-equilibrium',source]
            if execute(level+'_'+geometry,command):
                failures.append(level+'/'+geometry)
        coarse=root/'cg4'/geometry/'completed.json'
        fine=root/'cg6'/geometry/'completed.json'
        if coarse.exists() and fine.exists():
            destination=fine.parent/'refinement.json'
            if not destination.exists():
                command=[python,'-m','projects.diocotron.studies.equilibrium_catalog.check_refinement',
                    '--coarse',json.loads(coarse.read_text())['checkpoint'],
                    '--fine',json.loads(fine.read_text())['checkpoint'],
                    '--quadrature','8','12','--output',str(destination)]
                if execute('compare_cg4_cg6_'+geometry,command):failures.append('comparison/'+geometry)
        for level in ['cg4','cg6']:
            if any((root/level/g/'completed.json').exists() for g in ['star','uglypacman','iter','horseshoe']):
                execute('gallery_'+level,[python,str(CODE / 'render_gallery.py'),'--level',level])
        execute('summarize',[python,str(CODE / 'summarize.py')])
    write('high_order.completed.json',dict(status='complete' if not failures else 'partial',
        failures=failures,finished=time.time(),guiding_center_evolution_performed=False))
except BaseException as error:
    write('high_order.failed.json',dict(error=repr(error),time=time.time()))
    traceback.print_exc()
    raise
