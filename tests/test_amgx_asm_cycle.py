"""Independent dense checks for the external finest-level ASM experiment."""
import numpy as np
import pytest

from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only


@pytest.mark.parametrize('sweeps', (1, 2))
def test_asm_staging_matches_symmetric_coarse_correction(sweeps):
    with kernel_cache_only(True):
        from scripts.guiding_center.poisson.replay_poisson_asm import AsmCycle, patch_coloring_bound
        patches = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]])
        q, size = 2, 12
        rng = np.random.default_rng(70703)
        a = np.zeros((size, size))
        restrictions = []
        for patch in patches:
            indices = (patch[:, None]*q+np.arange(q)).ravel()
            r = np.eye(size)[indices]
            raw = rng.standard_normal((3*q, 3*q))
            a += r.T @ (raw@raw.T+np.eye(3*q)) @ r
            restrictions.append(r)
        c = sum(r.T@np.linalg.solve(r@a@r.T, r) for r in restrictions)
        colors, _ = patch_coloring_bound(patches, size//q)
        chol = np.linalg.cholesky(a)
        assert np.linalg.eigvalsh(chol.T@c@chol).max() <= colors+1e-12
        p = rng.standard_normal((size, 4))
        g = p@np.linalg.solve(p.T@a@p, p.T)
        omega = 1.8/colors

        class Action:
            def __init__(self, matrix):
                self.matrix = matrix
            def matvec(self, vector):
                return self.matrix@vector
            apply = matvec

        action = AsmCycle(Action(a), Action(c), Action(g), omega, sweeps)
        matrix = np.column_stack([action.apply(col) for col in np.eye(size)])
        s = omega*c
        accumulated = sum(np.linalg.matrix_power(np.eye(size)-s@a, j)@s for j in range(sweeps))
        f = np.eye(size)-a@accumulated
        reference = accumulated+accumulated.T-accumulated.T@a@accumulated+f.T@g@f
        np.testing.assert_allclose(matrix, reference, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(matrix, matrix.T, rtol=2e-13, atol=2e-14)
        assert np.linalg.eigvalsh(matrix).min() > 0
        b = rng.standard_normal(size)
        np.testing.assert_allclose(action.apply(b), matrix@b, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize('order', (2, 3))
@pytest.mark.parametrize('lower_fraction', (.125, .01))
def test_fixed_chebyshev_asm_stages_match_dense_polynomial_and_symmetric_cycle(order, lower_fraction):
    """A fixed Chebyshev smoother preserves the symmetric coarse correction."""
    with kernel_cache_only(True):
        from hdgfem.linalg.face_hp_multigrid import _chebyshev_richardson_weights
        from scripts.guiding_center.poisson.replay_poisson_asm import AsmCycle, patch_energy_bound

        patches = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]])
        q, size = 2, 12
        rng = np.random.default_rng(70719)
        a = np.zeros((size, size))
        restrictions = []
        for patch in patches:
            indices = (patch[:, None]*q+np.arange(q)).ravel()
            r = np.eye(size)[indices]
            raw = rng.standard_normal((3*q, 3*q))
            a += r.T @ (raw@raw.T+np.eye(3*q)) @ r
            restrictions.append(r)
        c = sum(r.T@np.linalg.solve(r@a@r.T, r) for r in restrictions)
        upper = 4.0
        assert patch_energy_bound(patches, patches, size//q) == upper
        a_factor = np.linalg.cholesky(a)
        assert np.linalg.eigvalsh(a_factor.T@c@a_factor).max() <= upper+1e-12
        p = rng.standard_normal((size, 4))
        g = p@np.linalg.solve(p.T@a@p, p.T)
        lower = lower_fraction*upper
        weights = _chebyshev_richardson_weights(order, lower, upper)
        assert len(weights) == order
        assert len(set(weights)) == order

        class RecordingAction:
            def __init__(self, matrix):
                self.matrix = matrix
                self.inputs = []

            def matvec(self, vector):
                self.inputs.append(vector.copy())
                return self.matrix@vector

            apply = matvec

        correction, coarse = RecordingAction(c), RecordingAction(g)
        action = AsmCycle(RecordingAction(a), correction, coarse,
                          omega=None, sweeps=order, weights=weights)
        actual = np.column_stack([action.apply(col) for col in np.eye(size)])

        # Independent matrix-polynomial oracle, without Richardson weights:
        # E = T_n((midpoint*I-C*A)/radius) / T_n(midpoint/radius).
        identity = np.eye(size)
        midpoint, radius = .5*(upper+lower), .5*(upper-lower)
        argument = (midpoint*identity-c@a)/radius
        previous, polynomial = identity, argument
        for _ in range(2, order+1):
            previous, polynomial = polynomial, 2*argument@polynomial-previous
        normalization = np.polynomial.chebyshev.chebval(
            midpoint/radius, [0.0]*order+[1.0],
        )
        error = polynomial/normalization
        smoothing = np.linalg.solve(a.T, (identity-error).T).T
        residual_map = identity-a@smoothing
        reference = (smoothing+smoothing.T-smoothing.T@a@smoothing
                     + residual_map.T@g@residual_map)
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(actual, actual.T, rtol=2e-12, atol=2e-13)
        assert np.linalg.eigvalsh(actual).min() > 0

        # Inspect intermediate residuals as well: commuting polynomial factors
        # make the final matrix insensitive to a mistaken post-stage order.
        correction.inputs.clear()
        coarse.inputs.clear()
        rhs = rng.standard_normal(size)
        solution = action.apply(rhs)
        expected_inputs = []
        staged = np.zeros_like(rhs)
        for weight in weights:
            residual = rhs-a@staged
            expected_inputs.append(residual)
            staged += weight*(c@residual)
        coarse_rhs = rhs-a@staged
        staged += g@coarse_rhs
        for weight in reversed(weights):
            residual = rhs-a@staged
            expected_inputs.append(residual)
            staged += weight*(c@residual)
        assert len(correction.inputs) == 2*order
        assert len(coarse.inputs) == 1
        np.testing.assert_allclose(correction.inputs, expected_inputs, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(coarse.inputs[0], coarse_rhs, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(solution, staged, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(solution, reference@rhs, rtol=2e-12, atol=2e-13)
