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
