"""Area-averaged rasters, streamed snapshot reads and encoder options of the movie helpers; no PDE solve."""
import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace


def _space():
    return DGSpace(rectangle_mesh(5, 4), 3, basis_type="dub_orth")


def test_raster_bounds_are_those_of_from_mesh():
    from hybridge.io.raster import RasterGeometry, raster_bounds

    mesh = _space().mesh
    for width, height in ((40, 40), (64, 30), (25, 70)):
        assert raster_bounds(mesh, width, height) == RasterGeometry.from_mesh(mesh, width, height).bounds


def test_area_map_with_one_sample_is_the_pixel_centre_map():
    from hybridge.io.raster import RasterGeometry, area_sampling_matrix

    space = _space()
    geometry = RasterGeometry.from_mesh(space.mesh, 37, 29)
    field = space.project_callable(lambda x, y: np.sin(3 * x) * np.cos(2 * y))
    matrix, mask = area_sampling_matrix(space, geometry.bounds, 37, 29)
    expected = geometry.sampling_matrix(space) @ np.asarray(field.coeffs).ravel()
    np.testing.assert_allclose(matrix @ np.asarray(field.coeffs).ravel(), expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(mask.ravel(), geometry.element_ids < 0)


def test_area_map_averages_the_subpixel_samples_inside_the_mesh():
    from hybridge.io.raster import area_sampling_matrix, raster_bounds, subpixel_geometries

    space = _space()
    field = space.project_callable(lambda x, y: x * x - y + x * y)
    coefficients = np.asarray(field.coeffs).ravel()
    bounds = raster_bounds(space.mesh, 31, 23)
    matrix, mask = area_sampling_matrix(space, bounds, 31, 23, supersample=3)
    total, counts = np.zeros(31 * 23), np.zeros(31 * 23)
    for geometry in subpixel_geometries(space.mesh, bounds, 31, 23, supersample=3):
        inside = geometry.element_ids >= 0
        total[inside] += (geometry.sampling_matrix(space) @ coefficients)[inside]
        counts += inside
    np.testing.assert_allclose(matrix @ coefficients, total / np.maximum(counts, 1), rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(mask.ravel(), counts == 0)
    assert mask.any() and not mask.all()  # the 4% margin of the bounds lies outside the mesh
    with pytest.raises(ValueError, match="positive integer"):
        area_sampling_matrix(space, bounds, 31, 23, supersample=0)


@pytest.mark.parametrize("compressed", (True, False))
def test_npz_slices_stream_one_member(tmp_path, compressed):
    from hybridge.io.records import iter_npz_slices

    rng = np.random.default_rng(3)
    density, other = rng.standard_normal((5, 7, 6)), rng.standard_normal((2, 3))
    path = tmp_path / "snapshots.npz"
    (np.savez_compressed if compressed else np.savez)(path, density=density, other=other, times=np.arange(5.))
    slices = list(iter_npz_slices(path, "density"))
    assert len(slices) == 5 and all(s.flags.c_contiguous for s in slices)
    np.testing.assert_array_equal(np.stack(slices), density)
    np.testing.assert_array_equal(np.stack(list(iter_npz_slices(path, "times"))), np.arange(5.))
    first = next(iter_npz_slices(path, "density"))  # an unfinished stream closes cleanly
    np.testing.assert_array_equal(first, density[0])


def test_movie_writer_encoder_options_and_metadata(tmp_path):
    import subprocess

    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    from hybridge.io.movie import MovieWriter

    path = tmp_path / "options.mp4"
    writer = MovieWriter(path, fps=10, crf=18, preset="slow", threads=1, metadata={"title": "ring, m = 9"})
    for value in (40, 200):
        frame = np.full((48, 64, 4), value, dtype=np.uint8)
        frame[..., 3] = 255
        writer.append(frame)
    writer.close()
    probe = subprocess.run([ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True,
                           text=True)
    assert "ring, m = 9" in probe.stderr
    assert ffmpeg.count_frames_and_secs(str(path))[0] == 2
    for options in (dict(crf=60), dict(crf=1.5), dict(threads=0)):
        with pytest.raises(ValueError):
            MovieWriter(tmp_path / "bad.mp4", **options)
