"""Explicit fixed-cycle AMG configuration must not inherit solve stopping."""
import copy
import pytest
from hdgfem.backends.advection_cuda import _amgx_config_for_solve


@pytest.mark.parametrize('cycles', (1, 2))
def test_fixed_amg_cycles_disable_residual_stopping_without_mutating_input(cycles):
    source = dict(config_version=2, solver=dict(solver='AMG', max_iters=17,
        tolerance=1e-6, monitor_residual=1, store_res_history=1))
    before = copy.deepcopy(source)
    actual = _amgx_config_for_solve(config=source, fixed_amg_cycles=cycles)['solver']
    assert actual['max_iters'] == cycles
    assert actual['monitor_residual'] == 0 and actual['store_res_history'] == 0
    assert source == before
    normal = _amgx_config_for_solve(config=source)['solver']
    assert normal['monitor_residual'] == normal['store_res_history'] == 1
    assert normal['max_iters'] == 17


@pytest.mark.parametrize('cycles', (0, -1, True, 1.5))
def test_fixed_cycles_reject_invalid_count(cycles):
    with pytest.raises(ValueError, match='positive integer'):
        _amgx_config_for_solve(config={'solver': {'solver': 'AMG'}}, fixed_amg_cycles=cycles)


def test_fixed_cycles_reject_a_krylov_solver():
    with pytest.raises(ValueError, match='requires an AMG solver'):
        _amgx_config_for_solve(config={'solver': {'solver': 'PCGF'}}, fixed_amg_cycles=1)
