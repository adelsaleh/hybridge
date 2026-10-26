"""MP4 recording of DG fields (host sampling, synthetic fields); no PDE solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace

pytest.importorskip("imageio_ffmpeg")


def test_recorder_writes_frames_with_fixed_and_per_frame_limits(tmp_path):
    from hybridge.io.recording import RasterMovieRecorder

    space = DGSpace(rectangle_mesh(4, 4), 2, basis_type="dub_orth")
    path = tmp_path / "fields.mp4"
    with RasterMovieRecorder(space, path, panels=[("a", dict(cmap="viridis", clim=(0., 1.))),
                                                  ("b", dict(cmap="RdBu_r", clim=(-1., 1.)))],
                             width=64, height=64, size=(400, 200), fps=5., device=False) as recorder:
        for k in range(3):
            field = space.project_callable(lambda x, y, k=k: np.sin(x + k) * np.cos(y))
            raster = recorder.sample(field)
            assert raster.shape == (64, 64) and raster.count() > 0
            recorder.frame((raster, raster), caption=f"frame {k}", limits=(None, (-2., 2.)))
        assert recorder.frames == 3
    assert path.stat().st_size > 0
    assert path.with_suffix(".png").exists()


def test_recorder_samples_fields_of_another_degree_on_its_mesh(tmp_path):
    from hybridge.io.recording import RasterMovieRecorder

    mesh = rectangle_mesh(4, 4)
    space, raised = DGSpace(mesh, 2, basis_type="dub_orth"), DGSpace(mesh, 3, basis_type="dub_orth")

    def exact(x, y):  # degree 2: both spaces hold it exactly
        return 0.3 + x - 2.0 * y + x * y

    with RasterMovieRecorder(space, tmp_path / "fields.mp4", panels=[("a", dict(clim=(0., 1.))),
                                                                      ("b", dict(clim=(0., 1.)))],
                             width=48, height=48, size=(400, 200), fps=5., device=False) as recorder:
        own, other = recorder.sample(space.project_callable(exact)), recorder.sample(raised.project_callable(exact))
        np.testing.assert_allclose(other.filled(0.0), own.filled(0.0), atol=1e-12)
        recorder.sample(raised.project_callable(exact))
        assert len(recorder._other_samplers) == 1  # one cached sampler per other space
        with pytest.raises(ValueError, match="same mesh"):
            recorder.sample(DGSpace(rectangle_mesh(4, 4), 3, basis_type="dub_orth").project_callable(exact))


def test_diocotron_movie_observer_normalizes_the_perturbation(tmp_path):
    from scripts.guiding_center.runtime.observers import MovieObserver, Observers

    space = DGSpace(rectangle_mesh(3, 3), 2, basis_type="dub_orth")
    equilibrium = space.project_callable(lambda x, y: 1 - x * x - y * y)
    observer = MovieObserver(tmp_path / "run.mp4", interval=0.5, dt=0.1, density_scale=1.0,
                              label="test", raster=48, fps=5.)
    observer_calls = []
    combined = Observers(observer, observer_calls.append, None)
    for step in range(1, 13):
        t = 0.1 * step
        snapshot = SimpleNamespace(
            time=t, space=space, accepted_density=space.project_callable(lambda x, y: 0.5 + 0 * x),
            poisson_result=SimpleNamespace(field=space.project_callable(
                lambda x, y, t=t: 1 - x * x - y * y + 1e-6 * np.exp(t) * x)),
            equilibrium_potential=equilibrium)
        combined(snapshot)
    observer.close()
    assert len(observer_calls) == 12
    # Frames at t = 0.5 and 1.0 (the first call builds the recorder at t = 0.1 >= 0).
    assert observer.recorder.frames == 3
    assert (tmp_path / "run.mp4").stat().st_size > 0
