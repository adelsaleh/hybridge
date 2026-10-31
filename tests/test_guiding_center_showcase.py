"""Exercise the introductory BDF2 loop with scalar stand-ins and fake resources.

Every mesh, projection, solver and visualization entry point is replaced before
the example runs. No numerical solver, time integrator, device or renderer runs.
"""
import inspect
from types import SimpleNamespace

import pytest

import hybridge as hdg
import hybridge.io as display
import hybridge.io.recording as recording
import hybridge.transport.positivity as positivity
from hybridge.solvers.advection_reaction import AdvectionReactionHDGOptions
from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGOptions
from scripts.guiding_center import showcase


@pytest.mark.parametrize('positive', [False, True])
@pytest.mark.parametrize('movie', [False, True])
@pytest.mark.parametrize('fail_poisson', [False, True])
def test_showcase_bdf2_api_history_rendering_and_cleanup(
        monkeypatch, tmp_path, positive, movie, fail_poisson):
    """Bind fake calls against real signatures; verify history and cleanup."""
    mesh, initial = object(), object()
    space = SimpleNamespace(constant=lambda value: value)
    domain = object()
    transported, poisson_sources, projected, frames, closed = [], [], [], [], []
    signatures = {
        'poisson': inspect.signature(hdg.DiffusionReactionHDGSolver.solve),
        'source': inspect.signature(hdg.DiffusionReactionHDGSolver.set_source),
        'transport': inspect.signature(hdg.AdvectionReactionHDGSolver.solve),
        'project': inspect.signature(positivity.DensityPositivityProjector.project),
        'live': inspect.signature(display.HolovizScalarPanels.update_fields),
        'sample': inspect.signature(recording.RasterMovieRecorder.sample),
        'frame': inspect.signature(recording.RasterMovieRecorder.frame),
    }

    def replace(owner, name, implementation):
        signature = inspect.signature(getattr(owner, name))

        def checked(*args, **kwargs):
            signature.bind(*args, **kwargs)
            return implementation(*args, **kwargs)

        monkeypatch.setattr(owner, name, checked)

    class Resource:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *error):
            closed.append(self.name)
            return False

    class Poisson(Resource):
        def set_source(self, source):
            signatures['source'].bind(self, source)
            self.source = source
            return self

        def solve(self):
            signatures['poisson'].bind(self)
            poisson_sources.append(self.source)
            count = len(poisson_sources)
            if fail_poisson and count == 2:
                raise RuntimeError('canned Poisson failure')
            return SimpleNamespace(field=100. + count, flux=10. * count)

    def poisson_factory(given_space, *, source, boundary_condition, **options):
        assert given_space is space and boundary_condition == 0.
        inspect.signature(DiffusionReactionHDGOptions).bind(**options)
        return Poisson('poisson').set_source(source)

    class Transport(Resource):
        def solve(self, **kwargs):
            signatures['transport'].bind(self, **kwargs)
            transported.append(kwargs)
            return SimpleNamespace(field=(3., 5.)[len(transported)-1])

    def transport_factory(given_space, **options):
        assert given_space is space
        inspect.signature(AdvectionReactionHDGOptions).bind(**options)
        return Transport('transport')

    class Projector:
        def project(self, field):
            signatures['project'].bind(self, field)
            projected.append(field)
            return field + .25, {}

    class Plot(Resource):
        def update_fields(self, fields, **kwargs):
            signatures['live'].bind(self, fields, **kwargs)
            frames.append((fields, kwargs))

        def sample(self, field):
            signatures['sample'].bind(self, field)
            return ('raster', field)

        def frame(self, rasters, **kwargs):
            signatures['frame'].bind(self, rasters, **kwargs)
            frames.append((rasters, kwargs))

    def profile(given_domain, **kwargs):
        assert given_domain is domain
        assert kwargs['strength_mode'] == ('positive' if positive else 'balanced')
        return initial

    def project(given_initial, given_space, **kwargs):
        assert given_initial is initial and given_space is space
        return 2.

    replace(hdg, 'gmsh_smooth_star_mesh', lambda *a, **k: mesh)
    replace(hdg, 'DGSpace', lambda *a, **k: space)
    replace(hdg, 'MeshDomain', lambda *a, **k: domain)
    replace(hdg, 'sample_gaussian_blob_field', profile)
    replace(hdg, 'project_callable', project)
    replace(hdg, 'perpendicular_vector_field', lambda flux: flux)
    replace(hdg, 'DiffusionReactionHDGSolver', poisson_factory)
    replace(hdg, 'AdvectionReactionHDGSolver', transport_factory)
    replace(positivity, 'DensityPositivityProjector', lambda *a, **k: Projector())
    replace(display, 'HolovizScalarPanels', lambda *a, **k: Plot('live'))
    replace(recording, 'RasterMovieRecorder', lambda *a, **k: Plot('movie'))
    args = showcase.parser().parse_args([
        '--case', 'positive-turbulence' if positive else 'vortex-gas',
        '--dt', '.1', '--steps', '2', '--every', '1',
        *(['--movie', str(tmp_path/'movie/out.mp4')] if movie else []),
    ])
    if fail_poisson:
        with pytest.raises(RuntimeError, match='canned Poisson failure'):
            showcase.run(args)
    else:
        showcase.run(args)

    initial_rho = 2.25 if positive else 2.
    first_rho = 3.25 if positive else 3.
    assert transported[0] == dict(source=initial_rho, beta=1., reaction=1.)
    assert poisson_sources[:2] == [initial_rho, first_rho]
    if not fail_poisson:
        assert transported[1]['source'] == pytest.approx((4*first_rho-initial_rho)/3)
        assert transported[1]['beta'] == pytest.approx(2.)
        assert transported[1]['reaction'] == 1.
        assert poisson_sources[2] == (5.25 if positive else 5.)
    assert projected == ([2., 3.] + ([] if fail_poisson else [5.]) if positive else [])
    assert len(frames) == (1 if fail_poisson else 3)
    expected_fields = (initial_rho, 101.)
    assert frames[0][0] == (tuple(('raster', value) for value in expected_fields) if movie else expected_fields)
    assert frames[0][1]['limits'] == (
        ((0., 22.), (0., .17561634457388905)) if positive else ((-18., 18.), (-.07, .07)))
    assert closed == ['movie' if movie else 'live', 'transport', 'poisson']
