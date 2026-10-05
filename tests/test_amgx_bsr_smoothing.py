"""Native BSR cycle regressions using the installed AMGX library; no build/JIT."""
import numpy as np
import pytest

from scripts.guiding_center.poisson.amgx_bsr_smoothing import (
    CYCLE_COST_OPTIONS, HostAmgxSystem, SCHEDULES, coupled_spd_chain, cycle_gate,
    smoothing_config,
)


@pytest.fixture(autouse=True)
def installed_amgx_only():
    pytest.importorskip('pyamgx')
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
    with kernel_cache_only(True):
        yield


@pytest.mark.parametrize('block_size', (2, 7))
@pytest.mark.parametrize('presweeps,postsweeps', SCHEDULES)
def test_l1_cycle_does_not_retain_previous_corrections(block_size, presweeps, postsweeps):
    result = cycle_gate('l1', presweeps, postsweeps, block_size=block_size)
    assert result['passed'], {k:v for k,v in result.items() if k not in ('config','log')}


@pytest.mark.parametrize('presweeps,postsweeps', SCHEDULES)
def test_block_jacobi_chebyshev_cycle_is_repeatable(presweeps, postsweeps):
    result = cycle_gate('block_jacobi', presweeps, postsweeps, block_size=7)
    assert result['passed'], {k:v for k,v in result.items() if k not in ('config','log')}


@pytest.mark.parametrize('cycle_options', (None, *CYCLE_COST_OPTIONS.values()),
    ids=('existing', *(f'cycle_cost_{name}' for name in CYCLE_COST_OPTIONS)))
def test_block_jacobi_chebyshev_solves_coupled_bsr_with_estimated_spectrum(cycle_options):
    matrix = coupled_spd_chain(7, 513, variable_basis=True)
    zero, reuse = (None, None) if cycle_options is None else cycle_options
    config = smoothing_config('block_jacobi', 1, 2, tolerance=1e-11, max_iters=120,
        zero_start_fastpath=zero, reuse_initial_preconditioner=reuse)
    config['solver']['preconditioner'].update(max_levels=3, dense_lu_num_rows=14)
    exact = np.random.default_rng(816).standard_normal(matrix.shape[0])
    rhs = matrix @ exact
    with HostAmgxSystem(matrix, config) as system:
        solution = system.apply(rhs)
        assert system.solver.status == 'success'
    assert np.linalg.norm(rhs-matrix@solution) <= 2e-11
    np.testing.assert_allclose(solution, exact, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('block_size', (2, 7))
def test_weighted_power_interval_covers_exact_block_scaled_spectrum(block_size):
    """Check the interval against a symmetric generalized dense eigensolve."""
    import re
    from scipy.linalg import eigh
    from hybridge.linalg.amgx.host import initialize_pyamgx_once

    matrix = coupled_spd_chain(block_size, 17, variable_basis=True)
    dense = matrix.toarray()
    diagonal = np.zeros_like(dense)
    for row in range(17):
        block = slice(row*block_size, (row+1)*block_size)
        diagonal[block, block] = dense[block, block]
    largest = eigh(dense, diagonal, eigvals_only=True)[-1]
    config = smoothing_config('block_jacobi', 1, 2)
    config['solver'] = config['solver']['preconditioner']['smoother']
    config['solver'].update(monitor_residual=0, max_iters=1)
    log = []
    amgx = initialize_pyamgx_once()
    amgx.register_print_callback(log.append)
    try:
        with HostAmgxSystem(matrix, config):
            pass
    finally:
        amgx.register_print_callback(lambda message: print(message, end=''))
    intervals = re.findall(
        r'Chebyshev power spectrum: rows=(\d+) block_size=(\d+) min=(\S+) max=(\S+) iterations=(\d+)',
        ''.join(log))
    assert len(intervals) == 1, ''.join(log)
    rows, size, lower, upper, iterations = intervals[0]
    assert (int(rows), int(size), int(iterations)) == (17, block_size, 128)
    lower, upper = float(lower), float(upper)
    assert largest <= upper <= 1.11*largest
    assert 0 < lower < upper
    assert np.isclose(lower, upper/8, rtol=1e-10)


@pytest.mark.parametrize('block_size', range(2, 11))
@pytest.mark.parametrize('backend', ('legacy', 'cusparse_generic'))
@pytest.mark.parametrize('dtype', (np.float32, np.float64), ids=('fp32', 'fp64'))
def test_cycle_cost_jacobi_zero_and_warm_starts_match_block_solves(block_size, backend, dtype):
    """Check real Jacobi updates, stale storage, and the second nonzero iterate."""
    from hybridge.linalg.amgx.host import pyamgx_supports_real_dtype

    if not pyamgx_supports_real_dtype(dtype):
        pytest.skip("installed PyAMGX accepts float64 only; FP32 needs the mode-aware "
                    "binding from scripts/dev/build_pyamgx_precision.py")
    matrix = coupled_spd_chain(block_size, 513, variable_basis=True).astype(dtype)
    diagonal = np.stack([
        matrix.data[start + np.flatnonzero(matrix.indices[start:end] == row)[0]]
        for row, (start, end) in enumerate(zip(matrix.indptr[:-1], matrix.indptr[1:]))
    ]).astype(np.float64)
    rng = np.random.default_rng(527)
    rhs, other, guess = rng.standard_normal((3, matrix.shape[0])).astype(dtype)
    poison = np.full(matrix.shape[0], np.nan, dtype=dtype)
    rtol, atol = (5e-5, 5e-6) if dtype == np.float32 else (2e-11, 2e-12)

    for sweeps in (1, 2):
        cold_expected = np.zeros(matrix.shape[0])
        warm_expected = guess.astype(np.float64)
        for _ in range(sweeps):
            cold_expected += .8*np.linalg.solve(
                diagonal, (rhs-matrix@cold_expected).reshape(-1, block_size, 1)).ravel()
            warm_expected += .8*np.linalg.solve(
                diagonal, (rhs-matrix@warm_expected).reshape(-1, block_size, 1)).ravel()
        for fast in (0, 1):
            config = dict(config_version=2, exception_handling=1, determinism_flag=1,
                device_mem_pool_enabled=0, solver=dict(
                    solver='BLOCK_JACOBI', max_iters=sweeps, monitor_residual=0,
                    relaxation_factor=.8, bsr_spmv_backend=backend,
                    block_jacobi_zero_start_fastpath=fast,
                    block_jacobi_use_fused_small_blocks=int(block_size in (2, 3, 5))))
            with HostAmgxSystem(matrix, config, dtype=dtype) as system:
                cold = system.apply(rhs, poison, zero_initial_guess=True)
                warm = system.apply(rhs, guess)
                system.apply(other)
                repeated = system.apply(rhs, guess, zero_initial_guess=True)
                zero = system.apply(np.zeros_like(rhs), poison, zero_initial_guess=True)
            np.testing.assert_allclose(cold, cold_expected, rtol=rtol, atol=atol)
            np.testing.assert_allclose(warm, warm_expected, rtol=rtol, atol=atol)
            np.testing.assert_allclose(repeated, cold_expected, rtol=rtol, atol=atol)
            np.testing.assert_array_equal(zero, np.zeros_like(rhs))


@pytest.mark.parametrize('order', (1, 2, 4))
@pytest.mark.parametrize('variant', ('zero', 'reuse', 'both'))
def test_cycle_cost_chebyshev_preserves_polynomial_and_history(order, variant):
    """The shortcuts must reproduce the installed legacy polynomial, including warm starts."""
    matrix = coupled_spd_chain(7, 65, variable_basis=True)
    b, c, guess = np.random.default_rng(982).standard_normal((3, matrix.shape[0]))
    outputs = []
    for name in ('legacy', variant):
        zero, reuse = CYCLE_COST_OPTIONS[name]
        config = smoothing_config('block_jacobi',
            zero_start_fastpath=zero, reuse_initial_preconditioner=reuse)
        config['solver'] = config['solver']['preconditioner']['smoother']
        # Common fixed bounds isolate application changes from setup estimation.
        config['solver'].update(max_iters=3, monitor_residual=0,
            chebyshev_polynomial_order=order, chebyshev_lambda_estimate_mode=3,
            cheby_min_lambda=.375, cheby_max_lambda=3., verbosity_level=0)
        with HostAmgxSystem(matrix, config) as system:
            outputs.append([
                system.apply(b), system.apply(b, guess), system.apply(c),
                system.apply(b), system.apply(np.zeros_like(b)),
            ])
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('presweeps,postsweeps', SCHEDULES)
@pytest.mark.parametrize('variant', tuple(CYCLE_COST_OPTIONS))
def test_cycle_cost_vcycle_remains_repeatable_and_linear(variant, presweeps, postsweeps):
    zero, reuse = CYCLE_COST_OPTIONS[variant]
    config = smoothing_config('block_jacobi', presweeps, postsweeps,
        zero_start_fastpath=zero, reuse_initial_preconditioner=reuse)
    result = cycle_gate(block_size=7, config=config)
    assert result['passed'], {k:v for k,v in result.items() if k not in ('config', 'log')}
    if presweeps == postsweeps:
        assert result['bilinear_symmetry_defect'] <= 1e-12
