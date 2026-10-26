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
phase='high_order_refinement'

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
lock=(root/'high_order_refinement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
write('high_order_refinement.driver.json',dict(pid=os.getpid(),pgid=os.getpgid(0),
    stop=f'kill -- -{os.getpgid(0)}',started=time.time(),scope='DOLFINx equilibrium refinement only; no HDG runs'))
failures=[]
try:
    # Wait for the current p-refinement driver to release its process lock.
    with (root/'high_order.lock').open('a') as previous:
        fcntl.flock(previous,fcntl.LOCK_EX)
        fcntl.flock(previous,fcntl.LOCK_UN)
    marker=root/'high_order.completed.json'
    if marker.exists():
        shutil.copy2(marker,root/'high_order.initial_completion.json')
    # Completed geometries skip; ITER retries with fixed auxiliary quadrature.
    if not all((root/'cg6'/g/'completed.json').exists() for g in ['star','uglypacman','iter','horseshoe']):
        subprocess.run([sys.executable,str(CODE / 'high_order_launcher.py')],check=True)
    for geometry in (sys.argv[1:] or ['uglypacman','star','horseshoe','iter']):
        parent=root/'cg6'/geometry
        if not (parent/'completed.json').exists():
            failures.append(geometry+': CG6 seed not available');continue
        source=json.loads((parent/'completed.json').read_text())['archive_id']
        inputs=root/'inputs'/'cg6_refined'/geometry
        if not (inputs/'config.json').exists():
            scale=3. if geometry=='iter' else 1.
            command=[python,'-m','projects.diocotron.studies.equilibrium_catalog.prepare_refined',
                '--archive',str(archive),'--equilibrium',source,
                '--mesh-size',str(.02 if geometry=='iter' else .006),
                '--band-contours',str(parent/'contours.npz'),
                '--band-mesh-size',str(.0025*scale),'--band-buffer',str(.003*scale),
                '--grading','.3','--corner-mesh-size',str(.00001*scale),
                '--degree','6','--quadrature-degree','32','--output',str(inputs)]
            if execute('prepare_cg6_refined_'+geometry,command):
                failures.append(geometry+': mesh preparation');continue
        mesh=json.loads((inputs/'mesh.msh.json').read_text())
        if mesh['num_cells']>300000:
            failures.append(geometry+': mesh exceeds 300000 triangles');continue
        output=root/'cg6_refined'/geometry
        wait_for_case(output)
        if not (output/'completed.json').exists():
            resume=(output/'branch'/'run.json').exists()
            if output.exists() and not resume:output.rename(output.with_name(geometry+'_attempt_'+str(int(time.time()))))
            command=[mpi,'-n','8','-bind-to','core',python,'-u','-m',
                'projects.diocotron.studies.equilibrium_catalog.run_distance',
                '--config',str(inputs/'config.json'),'--geometry',geometry,
                '--name',geometry+'_dt080_cg6_refined_20261016','--output',str(output),
                '--archive',str(archive),'--target-distance','.8','--distance-tolerance','2e-7']
            command+=['--resume'] if resume else ['--warm-equilibrium',source]
            if execute('cg6_refined_'+geometry,command):
                failures.append(geometry+': equilibrium refinement');continue
        destination=output/'refinement.json'
        if not destination.exists():
            command=[python,'-m','projects.diocotron.studies.equilibrium_catalog.check_refinement',
                '--coarse',json.loads((parent/'completed.json').read_text())['checkpoint'],
                '--fine',json.loads((output/'completed.json').read_text())['checkpoint'],
                '--quadrature','8','12','--output',str(destination)]
            if execute('compare_cg6_mesh_'+geometry,command):failures.append(geometry+': comparison')
        execute('summarize_refined',[python,str(CODE / 'summarize.py')])
    write('high_order_refinement.completed.json',dict(status='complete' if not failures else 'partial',
        failures=failures,finished=time.time(),guiding_center_evolution_performed=False))
except BaseException as error:
    write('high_order_refinement.failed.json',dict(error=repr(error),time=time.time()))
    traceback.print_exc()
    raise
