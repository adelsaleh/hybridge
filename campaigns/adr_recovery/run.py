#!/usr/bin/env python3
"""Bounded p2–p9 ADR recovery campaign, source-checked resume and reporting."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from campaigns.common.lifecycle import CampaignStop, stop_on_signals, terminate_group, release_device, log
from campaigns.common.protocol import host_provenance, Sampler

DEGREES=tuple(range(2,10))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    roots=('hybridge','campaigns/adr_recovery','campaigns/common')
    paths=[p for root in roots for p in (REPO/root).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.sbatch')]
    paths += [REPO/'tests'/name for name in ('test_adr_recovery_tensor.py','test_adr_recovery_cuda.py','test_adr_recovery_campaign.py')]
    files={str(p.relative_to(REPO)):sha(p) for p in paths}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return dict(head=head,files=files)


def plan(args):
    if not 300 <= args.seconds <= 8700: raise ValueError('Campaign budget must be 300–8700 seconds')
    if not 2 <= args.repeats <= 7: raise ValueError('Use 2–7 timed repetitions')
    if not 2 <= args.min_nx <= args.max_nx <= 1024: raise ValueError('Invalid mesh bounds')
    if not 512 <= args.chunk_limit <= 32768: raise ValueError('Invalid chunk bound')
    return dict(degrees=list(DEGREES),seconds=args.seconds,repeats=args.repeats,min_nx=args.min_nx,
        max_nx=args.max_nx,chunk_limit=args.chunk_limit,verbosity=3,gpus=1,slurm_walltime='02:30:00',
        assembly=['raw','compact','tensor'],threads=[32,64,128,256,512],
        solvers=['cublas','magma','fused candidates'],lifecycle=['reuse','rebuild'])


def freeze(out,configuration):
    identity=source_identity()
    digest=hashlib.sha256(json.dumps(dict(configuration=configuration,sources=identity),sort_keys=True).encode()).hexdigest()
    manifest=dict(configuration=configuration,sources=identity,sha256=digest)
    path=out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest: raise RuntimeError('Source/configuration changed: use a fresh output directory')
    else:
        out.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        changed=set(subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO,text=True).splitlines())
        changed.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines())
        for name in sorted(changed & set(identity['files'])):
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/name).read_bytes())
        (out/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=REPO))
    return manifest


def qualification_passed(path):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    return counts['tests']>=36 and not any(counts[k] for k in ('failures','errors','skipped')),counts


def run_child(command,log_path,timeout,heartbeat=None):
    # Workers own CUDA contexts. stdout is inherited so -u and verbosity 3
    # remain visible continuously in Slurm; a child tee retains the same log.
    with log_path.open('w') as stream:
        sampler=Sampler(log_path.with_suffix('.gpu.csv'));sampler.start()
        child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        shown=0
        deadline=time.monotonic()+timeout
        try:
            while child.poll() is None:
                if heartbeat is not None: heartbeat()
                if time.monotonic()>=deadline: raise TimeoutError(f'Worker exceeded {timeout:.0f}s')
                with log_path.open() as reader:
                    reader.seek(shown);new=reader.read();shown=reader.tell()
                if new: print(new,end='',flush=True)
                time.sleep(1)
            with log_path.open() as reader:
                reader.seek(shown);print(reader.read(),end='',flush=True)
            return child.returncode
        finally:
            terminate_group(child)
            sampler.finish()


def report(out):
    rows=[]
    for degree in DEGREES:
        path=out/f'p{degree}'/'result.json'
        if path.exists(): rows.append(json.loads(path.read_text()))
    lines=['# ADR recovery p2–p9 on H200','',
        'FP64 recovery of prescribed mixed fields and traces on distorted affine meshes. '
        'This is a recovery performance study, not a PDE accuracy or turbulence run.','',
        'Reuse retains recovery geometry and flux maps; rebuild reconstructs those per-element data '
        'on every call while retaining immutable reference-basis tables. Both refresh velocity, '
        'stabilization, diffusion-dependent primal matrices and RHS. '
        'Already reconstructed upstream ADR fields are inputs in both modes.','',
        'Assembly and solver candidates are measured on a representative chunk. The full mesh is then '
        'processed in bounded chunks with the raw reference and selected recovery paths. Full-mesh totals '
        'sum synchronized per-chunk samples and exclude input-fixture construction. Setup, JIT/warmups, '
        'instrumented phases and memory are recorded separately.','',
        '| p | Cells | Policy | Raw reference (s) | Optimized (s) | Speedup | Assembly | Solver | Flux cache |',
        '|---:|---:|---|---:|---:|---:|---|---|---|']
    for row in rows:
        if row['status']!='complete':
            lines.append(f"| {row['order']} | — | {row['status']} | — | — | — | — | — | — |")
            continue
        for policy in ('reuse','rebuild'):
            a=row['full_mesh_totals'][f'baseline/{policy}']['median_seconds']
            b=row['full_mesh_totals'][f'optimized/{policy}']['median_seconds']
            c=row['winners'][policy]['config']
            lines.append(f"| {row['order']} | {row['elements']:,} | {policy} | {a:.6f} | {b:.6f} | {a/b:.3f} | "
                         f"{c['assembly']} ({c['threads']}) | {c['solver']} | {c['cached_flux']} |")
    for row in rows:
        lines+=['',f"## p{row['order']}",'',f"Status: {row['status']}. Detailed data: [result.json](p{row['order']}/result.json).",'',
            '| Assembly | Threads | Median ms/chunk | Registers/thread | Local bytes/thread | Status |',
            '|---|---:|---:|---:|---:|---|']
        for candidate in row.get('assembly',[]):
            attrs=candidate.get('attributes',{})
            seconds=candidate.get('median_seconds')
            lines.append(f"| {candidate['engine']} | {candidate['threads']} | {1000*seconds if seconds is not None else '—'} | "
                         f"{attrs.get('num_regs','—')} | {attrs.get('local_size_bytes','—')} | {candidate['status']} |")
        lines+=['','| Solver | Median ms/chunk | Maximum scaled backward error | Status |','|---|---:|---:|---|']
        for candidate in row.get('solvers',[]):
            seconds=candidate.get('median_seconds')
            lines.append(f"| {candidate['variant']} | {1000*seconds if seconds is not None else '—'} | "
                         f"{candidate.get('backward_error','—')} | {candidate['status']} |")
        if row.get('chunks'):
            lines+=['','Per-phase totals below sum one separate instrumented wall-time sample per chunk. '
                    'Parent phase `flux.inputs_and_recovery` includes its nested phases; do not add those twice.','',
                    '| Path | Policy | Phase | Sum seconds |','|---|---|---|---:|']
            values={}
            for chunk in row['chunks']:
                for phase in chunk['profile']:
                    key=(chunk['variant'],chunk['policy'],phase['phase'])
                    values[key]=values.get(key,0.)+phase['wall_seconds']
            for (variant,policy,phase),seconds in sorted(values.items()):
                lines.append(f'| {variant} | {policy} | {phase} | {seconds:.6f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(degrees=rows),indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=os.environ.get('HYBRIDGE_CAMPAIGN_OUT'))
    parser.add_argument('--plan',action='store_true');parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--seconds',type=int,default=8700)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--min-nx',type=int,default=192);parser.add_argument('--max-nx',type=int,default=512)
    parser.add_argument('--chunk-limit',type=int,default=8192)
    args=parser.parse_args(); configuration=plan(args)
    if args.plan: print(json.dumps(configuration,indent=2));return
    if args.output is None: parser.error('--output or HYBRIDGE_CAMPAIGN_OUT is required')
    out=args.output.resolve()
    if args.report: report(out);return
    manifest=freeze(out,configuration)
    if args.freeze: print(json.dumps(dict(output=str(out),sha256=manifest['sha256'])));return
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU execution requires a Slurm allocation')
    budget_path=out/'budget.json'
    spent=json.loads(budget_path.read_text()).get('used_seconds',0.) if budget_path.exists() else 0.
    started=time.monotonic()
    deadline=started+args.seconds-spent
    def heartbeat():
        temp=budget_path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(used_seconds=spent+time.monotonic()-started,
            limit_seconds=args.seconds,job_id=os.environ.get('SLURM_JOB_ID')))+'\n')
        temp.replace(budget_path)
    if deadline-started<60: raise TimeoutError('Cumulative campaign time budget exhausted')
    forks={r:subprocess.check_output(['git','-C',str(REPO.parent/r),'rev-parse','HEAD'],text=True).strip() for r in ('AMGX','pyamgx')}
    host_provenance(out,source_identity=manifest['sha256'],configuration=configuration,forks=forks)
    (out/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    magma=os.environ.get('HYBRIDGE_MAGMA_LIBRARY')
    (out/'runtime.json').write_text(json.dumps(dict(magma_library=magma,
        magma_sha256=sha(magma) if magma and Path(magma).is_file() else None,
        environment={k:os.environ[k] for k in ('HYBRIDGE_PRECISION','HYBRIDGE_CUDA_TUNING',
            'HYBRIDGE_CUDA_TUNING_SECONDS','HYBRIDGE_CUDA_TUNING_CANDIDATES','HYBRIDGE_MAGMA_ROOT',
            'HYBRIDGE_AMGX_BUILD_ROOT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS',
            'SLURM_JOB_ID','SLURM_JOB_NAME','SLURM_JOB_GPUS') if k in os.environ},
        nvcc=subprocess.check_output(['nvcc','--version'],text=True)),indent=2)+'\n')
    failures=[]
    with stop_on_signals():
        qualification=out/'qualification.xml'
        if not qualification.exists() or not qualification_passed(qualification)[0]:
            command=[sys.executable,'-B','-m','pytest','-vv','-s','-p','no:cacheprovider','--maxfail=2',
                '--basetemp',str(out/'pytest'),'--junitxml',str(qualification),
                'tests/test_adr_recovery_tensor.py','tests/test_adr_recovery_cuda.py']
            status=run_child(command,out/'qualification.log',min(900,deadline-time.monotonic()),heartbeat)
            if status or not qualification_passed(qualification)[0]: raise RuntimeError('Recovery qualification failed')
        for index,degree in enumerate(DEGREES):
            if source_identity()!=manifest['sources']: raise RuntimeError('Sources changed during the campaign')
            target=out/f'p{degree}';done=target/'COMPLETE.json'
            if done.exists():
                record=json.loads(done.read_text())
                if record['source_identity']!=manifest['sha256'] or record['result_sha256']!=sha(target/'result.json'):
                    raise RuntimeError(f'Invalid completion marker p{degree}')
                continue
            # A failed prior attempt is evidence and is never overwritten by resume.
            if target.exists():
                archive=out/f'p{degree}-attempt-{time.time_ns()}'
                target.rename(archive)
            target.mkdir()
            remaining=deadline-time.monotonic()-60
            budget=min(1100,int(remaining/(len(DEGREES)-index)))
            if budget<30: raise TimeoutError('Overall campaign deadline reached')
            command=[sys.executable,'-B','-m','campaigns.adr_recovery.benchmark','--order',str(degree),
                     '--output',str(target),'--seconds',str(max(10,budget-15)),
                     '--repeats',str(args.repeats),'--min-nx',str(args.min_nx),'--max-nx',str(args.max_nx),
                     '--chunk-limit',str(args.chunk_limit)]
            log(f'Starting p{degree}, budget {budget}s, verbosity 3')
            try:
                status=run_child(command,target/'worker.log',budget,heartbeat)
                result=json.loads((target/'result.json').read_text()) if (target/'result.json').exists() else {}
                if status or result.get('status')!='complete': failures.append(degree)
                else: done.write_text(json.dumps(dict(source_identity=manifest['sha256'],result_sha256=sha(target/'result.json')))+'\n')
            except TimeoutError as exc:
                failures.append(degree);log(str(exc))
            finally:
                released=release_device(timeout=30)
                log('Device release: '+json.dumps(released))
                if released.get('released') is not True: raise RuntimeError('GPU release was not verified')
            report(out)
            heartbeat()
    rows=report(out)
    if failures or len(rows)!=8 or any(r['status']!='complete' for r in rows):
        (out/'INCOMPLETE.json').write_text(json.dumps(dict(failed_degrees=failures))+'\n')
        raise SystemExit(1)
    (out/'COMPLETE.json').write_text(json.dumps(dict(source_identity=manifest['sha256'],
        degrees=list(DEGREES),report_sha256=sha(out/'REPORT.md'),summary_sha256=sha(out/'summary.json')))+'\n')


if __name__=='__main__':
    try: main()
    except CampaignStop: raise SystemExit(130)
