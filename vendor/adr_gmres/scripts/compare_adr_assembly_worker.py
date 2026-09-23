#!/usr/bin/env python3
"""Isolated cross-checkout ADR assembly diagnostic; no global PDE solve.

Reuse each checkout's assemblers. For master raw CUDA, intercept the existing
solver boundary to obtain its completed CSR without running AMGX. Timings end
before canonical host CSR conversion; that cost is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backend', choices=['branch-numpy', 'branch-cupy', 'master-numpy', 'master-numba', 'master-raw'], required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--mesh', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--canonical-bernstein', action='store_true', help='Include conversion to common Bernstein host CSR in each timing')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.root = a.root.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.root))
    import numpy as np
    import scipy
    from scipy import sparse
    import hdgfem
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace
    assert Path(hdgfem.__file__).resolve().is_relative_to(a.root), hdgfem.__file__
    spec = importlib.util.spec_from_file_location('adr_comparison_cases', Path(__file__).with_name('adv_diff_rea_cases.py'))
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    kw, exact = cases.get_case(a.case)
    kw['beta'] = tuple(v if callable(v) else (lambda x, y, v=v: v + 0*x) for v in kw['beta'])
    space = DGSpace(rectangle_mesh(a.mesh, a.mesh), a.degree, basis_type='dub_orth', volume_quad_1d=2*a.degree+2, edge_quad_1d=2*a.degree+2)
    gpu = a.backend in ('branch-cupy', 'master-raw')
    cp = None
    if gpu:
        import cupy as cp
    if a.backend == 'master-numba':
        import numba
        numba.set_num_threads(a.threads)
    if a.backend.startswith('master') and a.backend != 'master-numpy' and not np.isscalar(kw['diffusion']):
        (a.output/'result.json').write_text(json.dumps({'status':'unsupported', 'reason':'master fused ADR requires constant scalar diffusion', **vars_for_json(a)}, indent=2))
        return

    def sync():
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()

    if a.backend.startswith('branch'):
        from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction, _csr
        def assemble():
            return assemble_advection_diffusion_reaction(space=space, backend='cupy' if gpu else 'numpy', **kw)
        def canonical(assembled):
            return _csr(assembled.system), assembled.system.rhs.ravel()
    else:
        from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data, assemble_numpy, _normal_flux
        trace_ref = space.trace_space('legacy-lagrange')
        def prepare():
            tau_adv = np.maximum(_normal_flux(kw['beta'], space, trace_ref), 0.)
            return prepare_adr_data(kw['source'], kw['reaction'], kw['beta'], space,
                diffusion=kw['diffusion'], advection_stabilization=tau_adv,
                diffusion_stabilization=1., trace_space=trace_ref)
        if a.backend == 'master-numpy':
            def assemble():
                return assemble_numpy(prepare(), kw['boundary_condition'], space,
                    diffusion=kw['diffusion'], trace_space=trace_ref)
        elif a.backend == 'master-numba':
            from hdgfem.backends.advection_diffusion_reaction_numba import assemble_projected_adr_trace_system_eliminated_numba
            def assemble():
                return assemble_projected_adr_trace_system_eliminated_numba(prepare(),
                    kw['boundary_condition'], space, diffusion=kw['diffusion'], trace_space=trace_ref)
        else:
            from unittest.mock import patch
            from hdgfem.backends.advection_diffusion_reaction_raw_cuda import assemble_projected_adr_trace_system_eliminated_raw_cuda
            from hdgfem.solvers.advection_diffusion_reaction import AdvectionDiffusionReactionHDGOptions
            class AssemblyCaptured(Exception):
                def __init__(self, assembly):
                    self.assembly = assembly
            def capture(assembly, **kwargs):
                raise AssemblyCaptured(assembly)
            options = AdvectionDiffusionReactionHDGOptions(diffusion=kw['diffusion'], solver='amgx', hdg_postprocess='none')
            def assemble():
                with patch('hdgfem.backends.advection_cuda.solve_reduced_system_amgx_device', capture):
                    try:
                        assemble_projected_adr_trace_system_eliminated_raw_cuda(
                            kw['source'], kw['beta'], kw['reaction'], kw['boundary_condition'], space,
                            prepared=prepare(), options=options, trace_space=trace_ref,
                            preparation_seconds=0., total_start=perf_counter())
                    except AssemblyCaptured as captured:
                        return captured.assembly
                raise RuntimeError('CUDA assembly did not reach the solver boundary')
        def canonical(assembled):
            if a.backend == 'master-raw':
                rhs = cp.asnumpy(assembled.rhs)
                return sparse.csr_matrix((cp.asnumpy(assembled.data), cp.asnumpy(assembled.indices), cp.asnumpy(assembled.indptr)), shape=(rhs.size,rhs.size)), rhs
            ts = assembled.trace_system
            return sparse.coo_matrix((ts.data,(ts.rows,ts.cols)),shape=(ts.rhs.size,ts.rhs.size)).tocsr(), ts.rhs

    comparison_face_basis = space.quad_data.bas1d_of_ref_edg_qds
    if a.canonical_bernstein and a.backend.startswith('master'):
        from hdgfem.core.space import _bernstein_edge_basis
        comparison_face_basis = _bernstein_edge_basis(a.degree, space.quad_data.quads_JGL)
        native_canonical = canonical
        def canonical(assembled):
            matrix, rhs = native_canonical(assembled)
            transform = np.linalg.lstsq(space.quad_data.bas1d_of_ref_edg_qds.T,
                comparison_face_basis.T, rcond=None)[0]
            global_transform = sparse.kron(sparse.eye(matrix.shape[0]//(a.degree+1)), transform, format='csr')
            return (global_transform.T @ matrix @ global_transform).tocsr(), global_transform.T @ rhs

    samples = []
    for rep in range(a.warmup+a.repeats):
        sync()
        start = perf_counter()
        assembled = assemble()
        sync()
        elapsed = perf_counter()-start
        convert_start = perf_counter()
        matrix, rhs = canonical(assembled)
        matrix.sum_duplicates()
        matrix.sort_indices()
        sync()
        conversion = perf_counter()-convert_start
        if rep >= a.warmup:
            samples.append({'assembly_seconds':elapsed,'host_csr_seconds':conversion,'total_seconds':elapsed+conversion})
        del assembled
    sparse.save_npz(a.output/'matrix.npz',matrix,compressed=False)
    np.save(a.output/'rhs.npy',rhs)
    np.savez(a.output/'geometry.npz', faces=space.mesh.loc2glob_edge,
        normals=space.mesh.normals, orientations=space.mesh.orientations,
        free_faces=space.mesh.int_edges_inds, volume_points=space.quad_data.Krf_quads,
        volume_weights=space.quad_data.Krf_w, face_weights=space.quad_data.weights_JGL,
        volume_basis=space.quad_data.bas_of_quads, face_basis=comparison_face_basis)
    asymmetry = sparse.linalg.norm(matrix-matrix.T)/sparse.linalg.norm(matrix)
    source_hash = hashlib.sha256()
    for path in sorted((a.root/'hdgfem').rglob('*.py')):
        source_hash.update(str(path.relative_to(a.root)).encode())
        source_hash.update(path.read_bytes())
    result = {**vars_for_json(a), 'status':'passed', 'source_sha256':source_hash.hexdigest(),
        'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'revision':subprocess.check_output(['git','-C',str(a.root),'rev-parse','HEAD'],text=True).strip(),
        'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__,
        'shape':matrix.shape, 'nnz':matrix.nnz,'samples':samples,
        'medians':{k:float(np.median([r[k] for r in samples])) for k in samples[0]},
        'relative_asymmetry':float(asymmetry), 'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'note':'RSS includes warmup/JIT/imports; assembly excludes mesh/space creation; host CSR conversion is separate. Raw CUDA is intercepted before solve.'}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('backend','case','mesh','degree','status','medians','relative_asymmetry')}),flush=True)


def vars_for_json(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


if __name__ == '__main__':
    main()
