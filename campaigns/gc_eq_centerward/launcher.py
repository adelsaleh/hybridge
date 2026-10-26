"""Detached, restartable sequential midpoint-continuation campaign."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
from pathlib import Path
import json,os,subprocess,sys,time,traceback
root=output_directory('equilibrium_centerward_20261015')
archive=ARTICLE / 'equilibria'
repo=EQUILIBRIUM_REPO
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLBACKEND='Agg')
python=FENICS_PYTHON
mpi=str(Path(python).parent/'mpiexec')
def write(path,data):
 p=root/path;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
write('driver.json',dict(pid=os.getpid(),pgid=os.getpgid(0),stop=f'kill -- -{os.getpgid(0)}',started=time.time()))
try:
 for geometry,factor in [('star',4.5),('uglypacman',4.0),('iter',4.5),('horseshoe',4.5)]:
  marker=root/f'{geometry}.complete.json'
  if marker.exists():
   record=json.loads(marker.read_text());checkpoint=Path(record['checkpoint'])
   if checkpoint.exists() and (archive/'entries'/f"{record['archive_id']}.json").exists():
    print(f'SKIP completed {geometry}',flush=True);continue
   raise RuntimeError(f'Incomplete artifacts for {geometry} marker')
  output=root/geometry
  if output.exists(): output.rename(root/f'{geometry}_attempt_{int(time.time())}')
  runner='run_contours' if geometry=='iter' else 'run'
  cmd=[mpi,'-n','8','-bind-to','core',python,'-u','-m','projects.diocotron.studies.equilibrium_catalog.'+runner,'--config',str(root/'inputs'/f'{geometry}.json'),'--geometry',geometry,'--name-prefix',geometry+'_centerward_20261015','--output',str(output),'--midpoint-factors',str(factor),'--backend','mumps','--archive',str(archive),'--warm-equilibrium',geometry+'_inward_20261012_01','--warm-midpoint-continuation']
  if geometry=='iter':cmd+=['--atlas-failure','PDE and three closed contours only; no single-center flow atlas asserted']
  started=time.monotonic(); samples=[]
  with (root/f'{geometry}_{int(time.time())}.log').open('w') as log:
   process=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
   ticks=os.sysconf('SC_CLK_TCK');previous={};last=time.monotonic();peak=0.;active=0
   while process.poll() is None:
    time.sleep(1);now=time.monotonic();current={}
    for p in Path('/proc').iterdir():
     if not p.name.isdigit():continue
     try:
      args=(p/'cmdline').read_bytes().split(b'\0')
      if str(output).encode() not in args or b'equilibrium_catalog.' not in b' '.join(args) or b'mpiexec' in args[0]:continue
      s=(p/'stat').read_text().rsplit(')',1)[1].split();current[p.name]=(int(s[11])+int(s[12]))/ticks
     except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    increments=[max(0,v-previous[k]) for k,v in current.items() if k in previous]
    peak=max(peak,sum(increments)/(now-last));active=max(active,sum(v>.02 for v in increments));previous=current;last=now
  write(f'{geometry}.execution.json',dict(command=cmd,exit_code=process.returncode,wall_seconds=time.monotonic()-started,peak_cpu_cores=peak,peak_active_ranks=active))
  if process.returncode:raise RuntimeError(f'{geometry} exited {process.returncode}')
  manifest=json.loads((output/'manifest.json').read_text());records=manifest['records']
  if len(records)!=1 or manifest['failures']:raise RuntimeError(f'{geometry} not accepted')
  write(f'{geometry}.complete.json',records[0])
  print('ACCEPTED '+json.dumps(records[0]),flush=True)
 for module,extra in [('render',[]),('compare',['--archive',str(archive),'--previous-family','inward_20261012'])]:
  subprocess.run([python,'-m','projects.diocotron.studies.equilibrium_catalog.'+module,str(root),*extra],cwd=repo,env=env,check=True)
 with (root/'archive_verification.json').open('w') as log:
  subprocess.run([python,'-m','projects.diocotron.dolfinx.equilibrium_archive','--archive',str(archive),'verify'],cwd=repo,env=env,stdout=log,check=True)
 write('completed.json',dict(status='complete',finished=time.time(),geometries=['star','uglypacman','iter','horseshoe']))
except BaseException as error:
 write('failed.json',dict(error=repr(error),time=time.time()));traceback.print_exc();raise
