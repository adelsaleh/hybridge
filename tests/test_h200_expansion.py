"""Degree coverage, source compatibility and CPU/GPU placement for the expansion."""
import ast
import inspect
from pathlib import Path

import pytest

from campaigns.h100_solvers.expanded import design

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def study():
    return design(ROOT / 'run_configs/adr_unified_l5/inventory.json')


def test_expanded_study_covers_all_degrees_pure_cases_and_configs(study):
    assert study['degrees'] == list(range(1, 10))
    assert {case['equation'] for case in study['cases']} == {'poisson', 'transport', 'adr'}
    assert study['counts']['configurations'] > 56
    sources = {source for config in study['solver_catalog'] for source in config['sources']}
    assert {str(path.relative_to(ROOT)) for path in (ROOT / 'configs/amgx').glob('*.json')} <= sources
    assert all(case['degrees'] == list(range(1, 10)) for case in study['cases'])
    assert study['previous_dossier_is_budget_cap'] is False
    assert study['launch_ready'] is False
    assert study['placement']['host']['partition'] == 'genoa'


def test_cg_not_scheduled_on_nonsymmetric_equations_and_cpu_inverse_is_staged(study):
    for config in study['solver_catalog']:
        solver = config['solver']
        outer = solver.get('amgx_config', {}).get('solver', {})
        if isinstance(outer, dict) and outer.get('solver') in {'CG', 'PCG', 'PCGF', 'FCG'}:
            assert config['equations'] == ['poisson']
        if solver.get('configuration', {}).get('local_solver') == 'cpu_inverse':
            assert config['requires_host_precompute'] is True
    assert any('ignores nested preconditioner' in note for config in study['solver_catalog'] for note in config['notes'])


def test_tensor_common_kwargs_compatibility_rejects_quadrature_point_coefficients(monkeypatch):
    from hybridge.transport import tsle_tensor
    function = tsle_tensor.assemble_projected_advection_trace_system_eliminated_tsle_tensor
    signature = inspect.signature(function)
    for name in ('beta_volume_values', 'beta_normal_flux', 'reaction_values'):
        assert signature.parameters[name].default is None
    arguments = {name: None for name, p in signature.parameters.items() if p.default is inspect.Parameter.empty}
    monkeypatch.setattr(tsle_tensor, 'require_cupy', lambda: None)
    for name in ('beta_volume_values', 'beta_normal_flux', 'reaction_values'):
        with pytest.raises(NotImplementedError,
                           match=r"quadrature-point coefficients; use raw_local_assembly='fused' or 'split3'"):
            function(**arguments, **{name: object()})


def test_native_adapter_uses_current_modules_without_importing_vendor_numerics():
    path = ROOT / 'campaigns/adr_unified/native_hp_worker.py'
    imports = [node.module for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ImportFrom)]
    assert 'hybridge.linalg.multigrid.krylov' in imports
    assert 'hybridge.linalg.gpu.production_gmres' in imports
    assert all(not name.startswith('hdgfem') for name in imports if name)


def test_recovery_cli_rejects_wrong_equation_cache_combinations(monkeypatch, tmp_path):
    from scripts.gpu import benchmark_hdg_recovery as benchmark
    for equation, policy in [('poisson', 'schur-lu+mass'), ('adr', 'schur-cholesky')]:
        monkeypatch.setattr('sys.argv', ['benchmark', '--equation', equation, '--order', '9',
                                       '--cache-policy', policy, '--output', str(tmp_path / 'result.json')])
        with pytest.raises(SystemExit):
            benchmark.main()
    assert not (tmp_path / 'result.json').exists()
