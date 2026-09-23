"""Algebraic regression gates for opt-in constant-vector BSR interpolation.

Requires a user-built AMGX containing block_graph_dense_constraint_mode=
constant_vector. The tests use the existing library and forbid compilation.
"""
import json

import numpy as np
import pytest

from scripts.guiding_center.poisson.amgx_bsr_smoothing import (
    ROOT, HostAmgxSystem, coupled_spd_chain, cycle_gate,
)
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only

CONFIG = ROOT / 'configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_constant_vector_bsr.json'


@pytest.fixture(autouse=True)
def existing_library_only():
    pytest.importorskip('pyamgx')
    with kernel_cache_only(True):
        yield


@pytest.mark.parametrize('block_size', (5, 6, 7))
@pytest.mark.parametrize('smoothing_steps', (1, 2))
def test_constant_vector_balanced_cycle_is_fixed_symmetric_and_positive(block_size, smoothing_steps):
    config = json.loads(CONFIG.read_text())
    config['solver']['preconditioner'].update(
        presweeps=2, postsweeps=2, block_graph_dense_smoothing_steps=smoothing_steps)
    result = cycle_gate(config=config, block_size=block_size)
    assert result['passed'], result
    assert result['bilinear_symmetry_defect'] < 1e-11, result
    assert result['sampled_energy_b'] > 0
    assert result['sampled_energy_c'] > 0


@pytest.mark.parametrize('dtype', (np.float64, np.float32))
def test_constant_vector_hierarchy_solves_spd_system_in_variable_basis(dtype):
    pyamgx = pytest.importorskip('pyamgx')
    if dtype == np.float32 and not getattr(pyamgx, 'HDGFEM_PRECISION_AWARE', False):
        pytest.skip('FP32 requires the existing precision-aware PyAMGX binding')
    matrix = coupled_spd_chain(7, variable_basis=True)
    config = json.loads(CONFIG.read_text())
    tolerance = 1e-10 if dtype == np.float64 else 1e-3
    config['solver'].update(tolerance=tolerance)
    config['solver']['preconditioner'].update(max_levels=4, dense_lu_num_rows=14)
    rng = np.random.default_rng(20260914)
    rhs, guess = rng.standard_normal((2, matrix.shape[0])).astype(dtype)
    with HostAmgxSystem(matrix, config, dtype=dtype) as system:
        actual = system.apply(rhs, guess)
        assert 0 < system.solver.iterations_number < 300
    # Check the original FP64 operator, not AMGX's recursive residual history.
    assert np.linalg.norm(rhs-matrix @ actual) < 5*tolerance
