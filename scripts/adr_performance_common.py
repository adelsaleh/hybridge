"""Pure planning, ranking and persistence helpers for the ADR performance campaign."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import math

FAMILIES = ('none','poly','block_jacobi','block_jacobi_poly','asm','asm_poly')
OPERATORS = ('raw','raw_fused','matmul')
INVERSES = ('cpu_inverse','gpu_inverse','cublas_inverse','gpu_solve')
ORTHOGONALIZATIONS = ('mgs','mgs2','cgs','cgs2')


@dataclass(frozen=True)
class Configuration:
    preconditioner: str = 'asm_poly'
    operator: str = 'raw_fused'
    application: str | None = 'fused'
    local_solver: str | None = 'cublas_inverse'
    polynomial_degree: int | None = 12
    orthogonalization: str = 'cgs2'
    restart: int = 75
    polynomial_orthogonalization: str | None = 'cgs2'

    @property
    def key(self):
        return fingerprint(asdict(self))[:16]

    @property
    def path(self):
        return (self.preconditioner,self.operator,self.application,self.local_solver)

    def gpu_options(self):
        return dict(preconditioner=self.preconditioner,operator=self.operator,
            asm_application=self.application if self.preconditioner.startswith('asm') else 'raw',
            block_jacobi_application=self.application if self.preconditioner.startswith('block_jacobi') else 'raw',
            local_solver=self.local_solver or 'cublas_inverse',
            polynomial_degree=self.polynomial_degree or 1,
            polynomial_setup_orthogonalization=self.polynomial_orthogonalization or 'cgs2',
            restart=self.restart,orthogonalization=self.orthogonalization,
            autotune=False,dtype='float64',raise_on_failure=False,
            # Pure CGS is a distinct candidate; do not silently replace it by CGS2.
            cgs2_fallback_threshold=None)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def path_configurations(families=FAMILIES,operators=OPERATORS,local_solvers=INVERSES,
                        *, degree=12,restart=75,orthogonalization='cgs2'):
    """Every supported implementation path; no duplicate irrelevant combinations."""
    configs=[]
    for family,op in product(families,operators):
        has_base=family.startswith(('asm','block_jacobi'))
        has_poly=family.endswith('poly')
        applications=('raw','matmul','fused') if family.startswith('asm') else ('raw','matmul')
        for inverse in local_solvers if has_base else (None,):
            for application in applications if has_base else (None,):
                if inverse=='gpu_solve' and application!='matmul':
                    continue
                configs.append(Configuration(family,op,application,inverse,degree if has_poly else None,
                    orthogonalization,restart,'cgs2' if has_poly else None))
    return list(dict.fromkeys(configs))


def parameter_grid(configs,*,degrees,restarts,orthogonalizations,polynomial_orthogonalizations):
    result=[]
    for c in configs:
        for d,r,o,po in product(degrees if c.polynomial_degree is not None else (None,),
                restarts,orthogonalizations,
                polynomial_orthogonalizations if c.polynomial_degree is not None else (None,)):
            result.append(replace(c,polynomial_degree=d,restart=r,orthogonalization=o,
                                  polynomial_orthogonalization=po))
    return list(dict.fromkeys(result))


def trace_dofs(n,p):
    return (3*n*n-2*n)*(p+1)


def mesh_for_target(p,target=1_000_000):
    n=(2+math.sqrt(4+12*target/(p+1)))/6
    return min((max(1,math.floor(n)),max(1,math.ceil(n))),key=lambda n:abs(trace_dofs(n,p)-target))


def estimates(n,p,restart=100):
    """Conservative planning estimates, not guarantees or measured peaks.

    Include full mixed matrices, quadrature temporaries, face systems, inverse
    workspaces and Arnoldi storage. Guards are checked again with live free RAM.
    """
    ne=2*n*n; P=(p+1)*(p+2)//2; F=p+1; nf=3*n*n+2*n
    mixed=ne*(3*P)**2*8
    boundary=ne*3*P*3*F*8
    elements=ne*9*F*F*8
    global_bytes=nf*5*F*F*8
    quadrature=ne*(2*p+2)**2*8
    krylov=trace_dofs(n,p)*(restart+8)*8
    return dict(trace_dofs=trace_dofs(n,p),elements=ne,
        assembly_host_bytes=6*mixed+6*boundary+4*elements+6*global_bytes+32*quadrature,
        assembly_device_bytes=4*mixed+5*boundary+4*elements+3*global_bytes,
        solver_device_bytes=8*global_bytes+8*elements+krylov,
        cache_disk_bytes=mixed+boundary+2*elements+global_bytes+ne*3*P*8)


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if hasattr(value,'item'):return json_safe(value.item())
    return value


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def eligible(row):
    samples=row.get('samples',[])
    return (row.get('status')=='passed' and len(samples)==row.get('requested_repeats')
            and bool(samples) and all(s.get('passed') for s in samples)
            and isinstance(row.get('setup_solve_median_ms'),(int,float))
            and math.isfinite(row['setup_solve_median_ms']))


def best_by(rows,group):
    """Select only fully measured, numerically validated configurations."""
    best={}
    for row in rows:
        if not eligible(row):continue
        key=group(row)
        if key not in best or row['setup_solve_median_ms']<best[key]['setup_solve_median_ms']:
            best[key]=row
    return list(best.values())


def flat_rows(rows):
    result=[]
    for r in rows:
        c=r.get('configuration',{})
        result.append({**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},**c,
            'measured_samples':len(r.get('samples',[]))})
    return result


def write_csv(path,rows):
    rows=flat_rows(rows)
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
