"""Small initial-field diagnostics; no PDE solves or time integration."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from hdgfem.core.geometry import DiskDomain, PolygonDomain
from hdgfem.cases.profiles import (
    FFTGaussianBlobField,
    GaussianBlobField,
    sample_gaussian_blob_field,
)
from hdgfem.runtime.precision import REAL_DTYPE
from scripts.guiding_center.cases.guiding_center_cases import positive_turbulence
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config, _validate_config


FFT_PRESET = "positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"
BASE_PRESET = "positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"


def source_field():
    return GaussianBlobField([[-0.21, 0.13], [0.19, -0.17], [0.05, 0.09]],
                             [0.06, 0.08, 0.10], [1.2, 0.7, 2.1], cutoff=4)


def fft_field(grid_shape=(129, 257)):
    return FFTGaussianBlobField(source_field(), bounds=((-0.8, -0.9), (0.8, 0.9)),
                                grid_shape=grid_shape, chunk_size=211)


def test_fft_reconstruction_converges_to_same_gaussians_and_preserves_mass():
    coarse, fine = fft_field((65, 129)), fft_field()
    x, y = np.meshgrid(np.linspace(-0.55, 0.55, 113), np.linspace(-0.55, 0.55, 117))
    expected = fine.source(x, y)
    errors = [np.linalg.norm(field(x, y)-expected)/np.linalg.norm(expected) for field in (coarse, fine)]
    assert errors[1] < 0.012
    assert errors[1] < 0.4*errors[0]
    for field in (coarse, fine):
        expected_mass = np.sum(field.strengths * 2*np.pi*field.sigmas**2
                               * (-np.expm1(-0.5*field.cutoff**2)))
        mass = field._host_grid.sum(dtype=np.float64) * np.prod(field.spacing)
        tolerance = 2e-6 if REAL_DTYPE == np.float32 else 2e-13
        assert mass == pytest.approx(expected_mass, rel=tolerance)
        assert np.all(field(x, y) >= 0)
        assert field(x, y).dtype == np.dtype(REAL_DTYPE)


def test_fft_field_has_compact_support_no_wraparound_and_caches_grid(monkeypatch):
    source = GaussianBlobField([[-0.69, 0.01]], [0.055], [3.0], cutoff=3)
    field = FFTGaussianBlobField(source, bounds=((-1, -1), (1, 1)), grid_shape=(129, 129))
    assert field._host_grid is None
    assert field(np.array([]), np.array([])).size == 0
    assert field._host_grid is None
    assert field(-0.69, 0.01).shape == ()
    assert field(-0.69, 0.01) > 2.8
    grid = field._host_grid

    def unexpected_rebuild(*args):
        pytest.fail("the FFT grid should be reused")

    monkeypatch.setattr(field, "_build_grid", unexpected_rebuild)
    y = np.linspace(-1, 1, 93)
    np.testing.assert_array_equal(field(np.full_like(y, 0.99), y), 0)
    np.testing.assert_array_equal(field(y+100, y), 0)
    assert field(y[:4, None], y[None, :7]).shape == (4, 7)
    assert field._host_grid is grid


@pytest.mark.parametrize("domain", [DiskDomain(), PolygonDomain([[-1, -1], [1, -1], [1, 1], [-1, 1]])])
def test_fft_sampler_preserves_empty_wall_band_after_reconstruction(domain):
    params = dict(counts=(7, 3), sigmas=(0.035, 0.06), cutoff=4,
                  wall_clearance=0.04, strength_mode="positive", fft_grid_shape=(129, 257))
    field = sample_gaussian_blob_field(domain, **params)
    repeated = sample_gaussian_blob_field(domain, **params)
    np.testing.assert_array_equal(field.centers, repeated.centers)
    np.testing.assert_array_equal(field.strengths, repeated.strengths)
    assert np.all(domain.boundary_distance(field.centers)
                  > 0.04 + field.cutoff*field.sigmas + field.support_padding)
    if isinstance(domain, DiskDomain):
        angle = np.arange(257)*2*np.pi/257
        radius = np.linspace(0.96, 1, 7)[:, None]
        x, y = radius*np.cos(angle), radius*np.sin(angle)
    else:
        edge = np.linspace(-1, 1, 257)
        near = np.linspace(0.96, 1, 7)[:, None] * np.ones_like(edge)
        x = np.concatenate((near.ravel(), -near.ravel(), np.tile(edge, 14)))
        y = np.concatenate((np.tile(edge, 14), near.ravel(), -near.ravel()))
    np.testing.assert_array_equal(field(x, y), 0)


@pytest.mark.parametrize("shape", [(0, 16), (16, 3), (32, 32.5), (32,), (8, 8)])
def test_fft_rejects_invalid_or_underresolved_grids(shape):
    with pytest.raises(ValueError, match="grid"):
        fft_field(shape)


def test_fft_rejects_signed_blobs_and_bad_bounds():
    with pytest.raises(ValueError, match="nonnegative"):
        FFTGaussianBlobField(GaussianBlobField([[0, 0]], 0.1, -1),
                             bounds=((-1, -1), (1, 1)), grid_shape=(129, 129))
    with pytest.raises(ValueError, match="bounds"):
        FFTGaussianBlobField(source_field(), bounds=((1, 1), (-1, -1)), grid_shape=(129, 129))
    with pytest.raises(ValueError, match="positive"):
        sample_gaussian_blob_field(DiskDomain(), (4,), (0.05,), fft_grid_shape=(129, 129))


def test_fft_case_and_response_file_keep_iter_solver_settings():
    args = build_parser().parse_args([f"@run_configs/guiding_center/{FFT_PRESET}.args"])
    assert args.preset == FFT_PRESET
    config = _runtime_config(preset_by_key(args.preset), args)
    _validate_config(config)
    base_args = build_parser().parse_args([BASE_PRESET])
    base = asdict(_runtime_config(preset_by_key(BASE_PRESET), base_args))
    differences = {key for key, value in asdict(config).items() if value != base[key]}
    assert differences <= {"case_params", "description", "diagnostics_prefix"}
    assert config.case_params == {**base["case_params"], "fft_grid_shape": (2048, 4096)}
    assert config.diagnostics_prefix == FFT_PRESET
    args = build_parser().parse_args([FFT_PRESET, "--case-param", "fft_grid_shape=[1024,2048]"])
    assert _runtime_config(config, args).case_params["fft_grid_shape"] == [1024, 2048]
    case = positive_turbulence(counts=(5,), sigmas=(0.04,), cutoff=4, fft_grid_shape=(129, 257))
    assert isinstance(case.initial_density, FFTGaussianBlobField)
    assert case.parameters["initial_profile"] == "fft_gaussian"
    assert case.parameters["fft_grid_shape"] == (129, 257)
    assert case.initial_density._host_grid is None  # Build only on the requested backend.
    assert FFT_PRESET in Path("scripts/guiding_center/example_runs.md").read_text()


def test_fft_gpu_matches_host_and_keeps_grid_and_projection_on_device():
    """User-run GPU check: CuPy may compile kernels on first use."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    from hdgfem.core.projection import project_callable
    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace

    host, device = fft_field(), fft_field()
    x, y = np.meshgrid(np.linspace(-0.7, 0.7, 29), np.linspace(-0.8, 0.8, 33))
    expected = host(x, y)
    actual = device(cp.asarray(x), cp.asarray(y))
    tolerance = 3e-5 if REAL_DTYPE == np.float32 else 3e-12
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance)
    assert bool(cp.all(actual >= 0))
    assert device._host_grid is None
    grid = device._device_grids[int(cp.cuda.Device().id)]
    device(cp.asarray(x), cp.asarray(y))
    assert device._device_grids[int(cp.cuda.Device().id)] is grid
    space = DGSpace(rectangle_mesh(2, 2, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5)), 2)
    expected = project_callable(host, space, volume_quad_1d=5)
    actual = project_callable(device, space, backend="device", volume_quad_1d=5)
    assert not actual.coefficients_materialized
    np.testing.assert_allclose(cp.asnumpy(as_cupy_coefficients(actual, as_cupy_space(space))),
                               expected.coeffs, rtol=tolerance, atol=tolerance)
    assert device._host_grid is None
