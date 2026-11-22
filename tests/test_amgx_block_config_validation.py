"""``_validate_amgx_block_configuration``: scalar-only AMGX components are refused for block matrices
before the native setup, with the fork's remedy in the message."""
import pytest

from hybridge.linalg.amgx.device_solver import _validate_amgx_block_configuration as validate


def _pcg(preconditioner):
    return {'config_version': 2, 'solver': {'solver': 'PCG', 'preconditioner': preconditioner}}


def test_plain_jacobi_l1_is_refused_for_block_rows():
    with pytest.raises(ValueError, match='JACOBI_L1 with block size 3.*jacobi_l1_scalar_rows_for_blocks'):
        validate(_pcg({'solver': 'JACOBI_L1'}), 3)
    with pytest.raises(ValueError, match='config.solver.preconditioner'):
        validate(_pcg('JACOBI_L1'), 2)


def test_jacobi_l1_passes_for_scalar_rows_or_with_the_fork_switch():
    validate(_pcg({'solver': 'JACOBI_L1'}), 1)
    validate(_pcg({'solver': 'JACOBI_L1', 'jacobi_l1_scalar_rows_for_blocks': 1}), 7)
    config = _pcg({'solver': 'JACOBI_L1'})
    config['solver']['jacobi_l1_scalar_rows_for_blocks'] = 1  # set in an enclosing scope
    validate(config, 7)


def test_jacobi_l1_smoother_of_a_scalar_expanded_hierarchy_passes():
    amg = {'solver': 'AMG', 'classical_bsr_hierarchy': 'scalar_expand', 'smoother': {'solver': 'JACOBI_L1'}}
    validate(_pcg(amg), 7)
    blocked = {'solver': 'AMG', 'smoother': {'solver': 'JACOBI_L1'}}
    with pytest.raises(ValueError, match='config.solver.preconditioner.smoother uses JACOBI_L1'):
        validate(_pcg(blocked), 7)


def test_block_capable_components_pass():
    validate({'solver': {'solver': 'PBICGSTAB', 'preconditioner': {'solver': 'BLOCK_JACOBI'}}}, 2)
    validate({'solver': {'solver': 'FGMRES', 'preconditioner': {'solver': 'MULTICOLOR_DILU'}}}, 10)
    with pytest.raises(ValueError, match='CHEBYSHEV_POLY'):
        validate({'solver': {'solver': 'PCG', 'preconditioner': {'solver': 'CHEBYSHEV_POLY'}}}, 2)
