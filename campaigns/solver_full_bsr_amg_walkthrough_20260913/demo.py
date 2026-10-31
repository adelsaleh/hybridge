"""Show the installed pure-BSR AMG hierarchy on a small algebraic problem.

No HDG simulation, time integration, CuPy kernel, or compilation is involved.
A compact scalar block graph is used to construct a coupled 7x7 BSR matrix.
The scalar coefficient matrix is never expanded into CSR.
"""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_output = _campaign_data_directory('hybridge/artifacts/full_bsr_amg_walkthrough_20260913')

import copy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.sparse import bsr_matrix, diags, eye, kron
import pyamgx

OUT = _campaign_output
ROOT = _campaign_workspace_path('hybridge')


def main():
    width, q = 64, 7
    one_d = diags((-np.ones(width-1), 2*np.ones(width), -np.ones(width-1)), (-1, 0, 1), format='csr')
    graph = (kron(eye(width, format='csr'), one_d, format='csr') + kron(one_d, eye(width, format='csr'), format='csr')).tocsr()
    graph.sort_indices()
    nf = graph.shape[0]
    rng = np.random.default_rng(20260913)
    coupling = rng.standard_normal((q, q))
    coupling = coupling @ coupling.T / q + np.eye(q)
    onsite = rng.standard_normal((q, q))
    onsite = onsite @ onsite.T / q + np.eye(q)
    values = np.ascontiguousarray(graph.data[:, None, None] * coupling[None])
    for row in range(nf):
        start, stop = graph.indptr[row:row+2]
        pos = start + np.flatnonzero(graph.indices[start:stop] == row)[0]
        values[pos] += 0.02 * onsite
    indptr = np.ascontiguousarray(graph.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(graph.indices, dtype=np.int32)
    A = bsr_matrix((values, indices, indptr), shape=(nf*q, nf*q))
    exact = rng.standard_normal(nf*q)
    rhs = np.ascontiguousarray(A @ exact)
    solution = np.zeros_like(rhs)

    config_path = ROOT / 'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_dense_bsr.json'
    config = json.loads(config_path.read_text())
    effective = copy.deepcopy(config)
    effective['solver'].update(max_iters=80, tolerance=1e-10)
    effective['solver']['preconditioner'].update(print_grid_stats=1, verbosity_level=3)
    (OUT/'effective_config.json').write_text(json.dumps(effective, indent=2)+'\n')
    log, objects = [], []
    pyamgx.initialize()
    pyamgx.register_print_callback(log.append)
    result = {}
    try:
        cfg = pyamgx.Config().create_from_dict(effective)
        objects.append(cfg)
        resource = pyamgx.Resources().create_simple(cfg)
        objects.append(resource)
        matrix = pyamgx.Matrix().create(resource, mode='dDDI')
        objects.append(matrix)
        matrix.upload(indptr, indices, values, block_dims=[q, q], shape=(nf, nf))
        b = pyamgx.Vector().create(resource, mode='dDDI')
        objects.append(b)
        x = pyamgx.Vector().create(resource, mode='dDDI')
        objects.append(x)
        b.upload_raw(rhs.ctypes.data, nf, q)
        x.upload_raw(solution.ctypes.data, nf, q)
        solver = pyamgx.Solver().create(resource, cfg, mode='dDDI')
        objects.append(solver)
        t0 = time.perf_counter()
        solver.setup(matrix)
        setup_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        solver.solve(b, x, zero_initial_guess=True)
        solve_seconds = time.perf_counter() - t0
        x.download(solution)
        residual = float(np.linalg.norm(rhs - A @ solution) / np.linalg.norm(rhs))
        error = float(np.linalg.norm(solution-exact)/np.linalg.norm(exact))
        result = {'kind':'Synthetic coupled block-matrix diagnostic; not an HDG mesh or performance benchmark.', 'scalar_block_graph_shape':[width,width], 'block_size':q, 'block_rows':nf, 'scalar_dimension':nf*q, 'fine_nonzero_blocks':len(values), 'amgx_status':solver.status, 'iterations':solver.iterations_number, 'true_relative_residual':residual, 'relative_solution_error':error, 'setup_seconds_diagnostic_only':setup_seconds, 'solve_seconds_diagnostic_only':solve_seconds, 'library_extension':pyamgx.__file__, 'config_file':str(config_path), 'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(), 'numerical_config_overrides':{'outer_max_iters':80, 'outer_tolerance':1e-10}, 'matrix_values_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
        assert solver.status == 'success' and residual < 1e-9, result
    finally:
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()
        (OUT/'amgx.log').write_text(''.join(log))
    levels = {}
    for line in ''.join(log).splitlines():
        match = re.match(r'\s*(\d+)\(D\)\s+(\d+)\s+(\d+)\s+', line)
        if match:
            level, scalar_rows, scalar_nnz = map(int, match.groups())
            assert scalar_rows % q == 0 and scalar_nnz % (q*q) == 0
            record = {'level':level, 'block_rows':scalar_rows//q, 'scalar_rows':scalar_rows, 'nonzero_blocks':scalar_nnz//(q*q), 'block_shape':[q,q]}
            assert level not in levels or levels[level] == record
            levels[level] = record
    assert len(levels) >= 2, 'Grid statistics were not captured'
    result['levels'] = [levels[i] for i in sorted(levels)]
    result['no_build_or_time_integration'] = True
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
