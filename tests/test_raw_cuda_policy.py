import pytest

from hybridge.hdg.cuda.launch import (
    recommended_raw_cuda_block_size,
    resolve_raw_cuda_block_size,
    triangle_element_dof,
)
from hybridge.solvers.advection_reaction import AdvectionReactionHDGOptions
from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGOptions


def test_solver_options_default_to_automatic_launch_selection():
    assert AdvectionReactionHDGOptions().raw_block_size == "auto"
    assert DiffusionReactionHDGOptions().raw_block_size == "auto"


@pytest.mark.parametrize(
    ("order", "expected"),
    ((0, 1), (1, 3), (2, 6), (6, 28), (8, 45), (9, 55)),
)
def test_triangle_element_dof(order, expected):
    assert triangle_element_dof(order) == expected


@pytest.mark.parametrize(
    ("order", "expected"),
    ((0, 32), (2, 32), (6, 32), (7, 64), (8, 128), (9, 128), (10, 128)),
)
def test_advection_recommendation_covers_local_rows(order, expected):
    assert recommended_raw_cuda_block_size("advection-reaction", order) == expected


@pytest.mark.parametrize(
    ("order", "expected"),
    ((0, 32), (2, 32), (3, 64), (4, 64), (5, 128), (6, 128)),
)
def test_diffusion_recommendation_uses_degree_tiers(order, expected):
    assert recommended_raw_cuda_block_size("diffusion-reaction", order) == expected


@pytest.mark.parametrize("order", range(10))
def test_adr_recommendation_extends_degree_tiers(order):
    """ADR's larger local solves do not change the legacy DR recommendation."""
    expected = 32 if order <= 2 else (64 if order <= 4 else 128)
    assert recommended_raw_cuda_block_size("advection-diffusion-reaction", order) == expected


def test_resolver_preserves_explicit_sizes_and_resolves_auto():
    assert resolve_raw_cuda_block_size(
        "auto", equation="diffusion-reaction", order=4
    ) == 64
    assert resolve_raw_cuda_block_size(
        None, equation="advection-reaction", order=6
    ) == 32
    assert resolve_raw_cuda_block_size(
        1, equation="diffusion-reaction", order=6
    ) == 1
    assert resolve_raw_cuda_block_size(
        "64", equation="advection-reaction", order=2
    ) == 64


@pytest.mark.parametrize("requested", (True, 32.0, 0, 16, 256, "default"))
def test_resolver_rejects_invalid_explicit_sizes(requested):
    with pytest.raises(ValueError, match="raw-CUDA block size"):
        resolve_raw_cuda_block_size(
            requested, equation="advection-reaction", order=2
        )


def test_recommendation_rejects_unqualified_orders():
    with pytest.raises(ValueError, match="no raw-CUDA block-size recommendation"):
        recommended_raw_cuda_block_size("diffusion-reaction", 7)
    with pytest.raises(ValueError, match="no raw-CUDA block-size recommendation"):
        recommended_raw_cuda_block_size("advection-reaction", 15)
    with pytest.raises(ValueError, match="no raw-CUDA block-size recommendation"):
        recommended_raw_cuda_block_size("advection-diffusion-reaction", 10)


def test_auto_local_assembly_is_memory_aware_from_p6():
    """split3 from p=8 always, from p=6 when its workspace fits a quarter of free memory."""
    from hybridge.solvers.capabilities import (
        RAW_SPLIT3_WORKSPACE_FRACTION,
        resolve_raw_local_assembly,
        split3_memory_check_needed,
    )

    eligible = dict(bsr_output=True, raw_lu_mode=None, cache_operator=False)
    assert resolve_raw_local_assembly("auto", order=8, **eligible) == "split3"
    assert resolve_raw_local_assembly("auto", order=6, **eligible) == "fused"  # no memory sizes given
    fits = dict(split3_workspace_bytes=1, available_device_bytes=int(1 / RAW_SPLIT3_WORKSPACE_FRACTION))
    tight = dict(split3_workspace_bytes=2, available_device_bytes=int(1 / RAW_SPLIT3_WORKSPACE_FRACTION))
    for order in (6, 7):
        assert resolve_raw_local_assembly("auto", order=order, **eligible, **fits) == "split3"
        assert resolve_raw_local_assembly("auto", order=order, **eligible, **tight) == "fused"
        assert split3_memory_check_needed("auto", order=order, **eligible)
    assert resolve_raw_local_assembly("auto", order=5, **eligible, **fits) == "fused"
    # split3 requirements still apply: no BSR output, safe LU or operator cache keep fused.
    for blocker in (dict(bsr_output=False), dict(raw_lu_mode="safe"), dict(cache_operator=True)):
        options = {**eligible, **blocker}
        assert resolve_raw_local_assembly("auto", order=6, **options, **fits) == "fused"
        assert resolve_raw_local_assembly("auto", order=8, **options) == "fused"
        assert not split3_memory_check_needed("auto", order=6, **options)
    assert not split3_memory_check_needed("auto", order=8, **eligible)
    assert not split3_memory_check_needed("fused", order=6, **eligible)
    assert resolve_raw_local_assembly("fused", order=8, **eligible) == "fused"
    assert resolve_raw_local_assembly("split3", order=2, bsr_output=True) == "split3"


def test_split3_workspace_bytes_matches_the_workspace_layout():
    from hybridge.runtime.precision import REAL_ITEMSIZE
    from hybridge.transport.tsle_bsr import split3_workspace_bytes

    # p=6: NEL=28, NTR=7 (NCOLS=22), NQF=8.
    assert split3_workspace_bytes(1000, 28, 7, 8) == 1000 * (28 * 28 + 28 * 22 + 6 * 8) * REAL_ITEMSIZE
