"""Block-DILU regressions using an independent Kronecker-factor oracle.

Requires the user's AMGX rebuild. These tests never build or compile kernels.
"""
import copy
import json

import numpy as np
import pytest
from scipy import sparse

from scripts.guiding_center.poisson.amgx_bsr_smoothing import ROOT, HostAmgxSystem, coupled_spd_chain, cycle_gate
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only


def dilu_config():
    return dict(config_version=2, exception_handling=1, determinism_flag=1,
        solver=dict(solver='MULTICOLOR_DILU', max_iters=1, relaxation_factor=1.,
            monitor_residual=0, coloring_level=1, matrix_coloring_scheme='PARALLEL_GREEDY',
            reorder_cols_by_color=0, insert_diag_while_reordering=0))


@pytest.fixture(autouse=True)
def existing_library_only():
    pytest.importorskip('pyamgx')
    with kernel_cache_only(True):
        yield


@pytest.mark.parametrize('block_size', (4, 7, 8, 10))
@pytest.mark.parametrize('block_rows,bandwidth', ((1024, 1), (128, 20)))
@pytest.mark.parametrize('dtype', (np.float64, np.float32))
def test_block_dilu_matches_kronecker_factor_oracle(block_size, block_rows, bandwidth, dtype):
    pyamgx = pytest.importorskip('pyamgx')
    if dtype == np.float32 and not getattr(pyamgx, 'HYBRIDGE_PRECISION_AWARE', False):
        pytest.skip('FP32 requires the existing precision-aware PyAMGX binding')
    n, q = block_rows, block_size
    offsets = list(range(-bandwidth, bandwidth+1))
    scalar = sparse.diags([np.full(n-abs(k), 2*bandwidth+.3 if k == 0 else -1.)
                          for k in offsets], offsets, shape=(n,n), format='csr')
    scalar.sort_indices()
    rng = np.random.default_rng(31904+q)
    raw = rng.standard_normal((q,q))
    coupling = raw @ raw.T/q + np.eye(q)
    rhs, other, guess = rng.standard_normal((3,n,q)).astype(dtype)
    guess *= .1
    # For A=G tensor C, block DILU is DILU(G) tensor inv(C). The scalar
    # applications plus an independent dense solve provide the oracle.
    scalar_bsr = scalar.tobsr(blocksize=(1,1))
    with HostAmgxSystem(scalar_bsr, dilu_config(), dtype=dtype) as system:
        scalar_corrections = np.column_stack([system.apply(rhs[:,j].copy()) for j in range(q)])
    oracle = np.linalg.solve(coupling, scalar_corrections.T).T.ravel()
    matrix = sparse.bsr_matrix((np.ascontiguousarray(scalar.data[:,None,None]*coupling),
                                scalar.indices, scalar.indptr), shape=(n*q,n*q))
    with HostAmgxSystem(matrix, dilu_config(), dtype=dtype) as system:
        first = system.apply(rhs.ravel())
        second = system.apply(other.ravel())
        repeated = system.apply(rhs.ravel())
        warm = system.apply(rhs.ravel(), guess.ravel())
        warm_oracle = guess.ravel() + system.apply(rhs.ravel()-matrix @ guess.ravel())
    tolerance = 5e-11 if dtype == np.float64 else 5e-5
    relative = lambda actual, expected: np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300)
    assert relative(first, oracle) < tolerance
    assert relative(repeated, first) < tolerance
    assert relative(warm, warm_oracle) < tolerance
    b, c = rhs.ravel().astype(float), other.ravel().astype(float)
    symmetry = abs(b@second-c@first)/(np.linalg.norm(b)*np.linalg.norm(second)+np.linalg.norm(c)*np.linalg.norm(first))
    assert symmetry < tolerance
    assert b @ first > 0 and c @ second > 0


@pytest.mark.parametrize('constraint', ('additive', 'constant_vector'))
def test_q7_dilu_balanced_bsr_cycle_is_fixed_symmetric_and_solves(constraint):
    config = json.loads((ROOT/'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json').read_text())
    amg = config['solver']['preconditioner']
    amg.update(presweeps=1, postsweeps=1, block_graph_dense_constraint_mode=constraint,
               coloring_level=1, matrix_coloring_scheme='PARALLEL_GREEDY')
    amg['smoother'] = copy.deepcopy(dilu_config()['solver'])
    gate = cycle_gate(config=config)
    assert gate['passed'], gate
    assert gate['bilinear_symmetry_defect'] < 1e-11, gate
    assert gate['sampled_energy_b'] > 0 and gate['sampled_energy_c'] > 0
    matrix = coupled_spd_chain(7, variable_basis=True)
    config['solver'].update(tolerance=1e-10)
    amg.update(max_levels=4, dense_lu_num_rows=14)
    rhs = np.random.default_rng(704).standard_normal(matrix.shape[0])
    with HostAmgxSystem(matrix, config) as system:
        actual = system.apply(rhs)
    assert np.linalg.norm(rhs-matrix@actual) < 5e-10
