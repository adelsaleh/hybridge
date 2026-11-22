"""Run by tests/test_adr_h200_tuned_device_checks.py in its own process (the vendored ``scripts``
package must shadow master's): device checks and the device AMGX BSR, executed with NumPy as
the array module, against the vendored host verify and SciPy BSR on a small assembled system."""
import sys
import types
from pathlib import Path

import numpy as np

root, cache = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root/'vendor/adr_gmres'))
sys.path.append(str(root))

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scripts.adr_performance_worker import get_space, verify, save_cache, load_cache
from scripts.adv_diff_rea_cases import get_case
from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction, _csr
from campaigns.adr_h200_tuned.device_verify import DeviceVerifier, save_geometry
from campaigns.adr_h200_tuned.device_measure import amgx_bsr_on_device
from hybridge.linalg.amgx.device_solver import _validate_bsr_values


class _Device:
    def synchronize(self):
        pass


shim = types.SimpleNamespace(**{name: getattr(np, name) for name in dir(np) if not name.startswith('_')})
shim.cuda = types.SimpleNamespace(Device=_Device, runtime=types.SimpleNamespace(memGetInfo=lambda: (1 << 40, 1 << 40)))
shim.asnumpy = np.asarray

for case, p in (('trigonometric', 2), ('variable_velocity', 3), ('anisotropic', 4)):
    spec = dict(n=5, p=p, case=case, rtol=1e-10, cache=str(cache/f'{case}_{p}'), mesh_path=None)
    space = get_space(spec)
    kw, exact = get_case(case)
    assembly = assemble_advection_diffusion_reaction(space=space, backend='numpy', **kw)
    save_cache(Path(spec['cache']), assembly, space)
    save_geometry(spec['cache'], space)
    system, get = load_cache(spec['cache'])
    solution = spsolve(_csr(system), np.asarray(system.rhs).reshape(-1))
    checker = DeviceVerifier.from_cache(shim, spec, system, get, exact, budget_bytes=1 << 40, log=lambda *_: None)
    assert checker.resident and checker.exact_on_device
    for x in (solution, solution*(1+1e-6*np.random.default_rng(1).standard_normal(solution.size))):
        host = verify(spec, space, system, get, x)
        device = checker.verify(x)
        assert host['passed'] == device['passed'], (case, host, device)
        # Residuals agree to round-off of ||b|| (a direct solution's residual is itself round-off);
        # the L2 error agrees to 1e-12 relative, or to round-off where the exact solution lies in
        # the discrete space (variable_velocity is quadratic).
        bn = float(np.linalg.norm(np.asarray(system.rhs)))
        assert abs(host['true_residual_norm']-device['true_residual_norm']) <= 1e-13*bn, (case, host, device)
        assert abs(host['true_relative_residual']-device['true_relative_residual']) <= 1e-13, (case, host, device)
        assert abs(host['l2_error']-device['l2_error']) <= max(1e-12*host['l2_error'], 1e-15), (case, host['l2_error'], device['l2_error'])
        assert abs(checker.relative_residual(x, system.rhs.reshape(-1))-host['true_relative_residual']) <= 1e-13
    valid = system.neighbors >= 0
    expected = sparse.bsr_matrix((np.ascontiguousarray(system.blocks[valid]), system.neighbors[valid].astype(np.int32),
                                  np.r_[0, np.cumsum(valid.sum(axis=1))].astype(np.int32)), shape=(system.num_dofs, system.num_dofs))
    expected.sort_indices()
    indptr, indices, data = amgx_bsr_on_device(shim, np.asarray(system.blocks), np.asarray(system.neighbors))
    assert np.array_equal(indptr, expected.indptr) and np.array_equal(indices, expected.indices)
    assert np.array_equal(data, expected.data)

    # The wrapper that hands the BSR to pyamgx refuses a flat value vector (pyamgx takes nnz from len(data)).
    view = types.SimpleNamespace(indptr=indptr, indices=indices, data=data, shape=(system.num_dofs,)*2, block_size=system.block_size)
    _validate_bsr_values(view, system.block_size)
    assert data.shape == (expected.nnz//system.block_size**2, system.block_size, system.block_size)
    try:
        _validate_bsr_values(types.SimpleNamespace(indptr=indptr, indices=indices, data=data.reshape(-1),
                                                   shape=view.shape, block_size=system.block_size), system.block_size)
        raise AssertionError('a flat values view must be refused')
    except ValueError:
        pass
    print(case, p, 'device checks and BSR match the vendored host code')
