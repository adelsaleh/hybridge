"""FP64-rich assembly priors of the raw-CUDA Poisson tuner (no GPU needed)."""
from types import SimpleNamespace

import pytest

from hybridge.mixed.raw_cuda import poisson


def _context(order, policy='schur-lu'):
    nel, ntr = (order + 1) * (order + 2) // 2, order + 1
    return SimpleNamespace(cspace=SimpleNamespace(el_dof=nel, edg_dof=ntr), policy=policy)


def _key(config):
    return config.threads, config.columns, int(config.retained_coupling)


# H200 NVL thorough winners at 100,352 triangles (dr-01, 2026-10-28), per order and policy.
H200_WINNERS = {
    (1, 'none'): (32, 7, 0), (2, 'none'): (32, 10, 1), (3, 'none'): (32, 13, 1), (4, 'none'): (32, 16, 1),
    (5, 'none'): (64, 19, 1), (6, 'none'): (128, 22, 1), (7, 'none'): (128, 25, 0), (8, 'none'): (128, 28, 0),
    (9, 'none'): (256, 31, 0), (2, 'schur-cholesky'): (64, 10, 1), (5, 'schur-cholesky'): (128, 19, 1),
    (7, 'schur-cholesky'): (256, 16, 1), (8, 'schur-cholesky'): (256, 16, 1), (9, 'schur-cholesky'): (256, 31, 0),
}


@pytest.mark.parametrize('order,policy', sorted(H200_WINNERS))
def test_fp64_rich_priors_rank_the_h200_winner_within_the_quick_four(order, policy):
    ranked = poisson.configurations(_context(order, policy), 0, {'fp32_fp64_ratio': 2})
    assert H200_WINNERS[order, policy] in [_key(c) for c in ranked[:4]]


def test_generic_ranking_is_unchanged_elsewhere():
    for device, phase, explicit in (({'fp32_fp64_ratio': 64}, 0, None), ({'fp32_fp64_ratio': None}, 0, None),
                                    ({'fp32_fp64_ratio': 2}, 1, None), ({'fp32_fp64_ratio': 2}, 0, 128)):
        ranked = poisson.configurations(_context(9), phase, device, explicit)
        assert not poisson._fp64_rich_priors_apply(phase, device, explicit)
        assert _key(ranked[0])[0] == (explicit or 64)
