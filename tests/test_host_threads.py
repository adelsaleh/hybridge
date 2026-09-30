"""Element-chunked host threading: chunking, nesting, errors, and threaded == serial results."""
import numpy as np
import pytest

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.assembly import matrices_numpy as mats
from hdgfem.runtime import threads as ht


@pytest.fixture
def four_threads():
    previous = ht.set_host_threads(4)
    yield
    ht.set_host_threads(previous)


def test_element_chunks_cover_the_range_once():
    for count, kwargs in ((1000, dict(threads=24)), (100, dict(threads=24)),
                          (1000, dict(min_chunk=10, max_chunk=100, threads=4)), (0, dict(threads=4))):
        chunks = ht.element_chunks(count, **kwargs)
        covered = np.concatenate([np.arange(a, b) for a, b in chunks]) if chunks else np.empty(0)
        np.testing.assert_array_equal(covered, np.arange(count))
    assert ht.element_chunks(1000, min_chunk=256, threads=24) == [(0, 333), (333, 666), (666, 1000)]
    assert len(ht.element_chunks(1000, min_chunk=10, max_chunk=100, threads=4)) == 10


def test_elementwise_matches_direct_with_nesting_components_and_scalars(four_threads):
    x = np.linspace(0., 1., 400*100).reshape(400, 100)
    inner = lambda a: ht.elementwise(np.sin, a, min_chunk=100, max_chunk=1000)
    np.testing.assert_array_equal(ht.elementwise(lambda a: 2.*inner(a), x, min_chunk=100, max_chunk=2000),
                                  2.*np.sin(x))
    stacked = ht.elementwise(lambda a, b: np.stack((a, b*a), -1), x, 3., min_chunk=100, max_chunk=5000)
    np.testing.assert_array_equal(stacked, np.stack((x, 3.*x), -1))
    assert ht.elementwise(np.sin, .5) == np.sin(.5)


def test_chunk_errors_propagate_after_every_chunk(four_threads):
    seen = []

    def failing(start, stop):
        seen.append(start)
        if start > 0:
            raise RuntimeError("chunk failed")

    with pytest.raises(RuntimeError, match="chunk failed"):
        ht.for_element_chunks(failing, 2000, min_chunk=100)
    assert sorted(seen) == [a for a, _ in ht.element_chunks(2000, min_chunk=100)]


def test_parallel_copy_is_a_contiguous_copy(four_threads):
    source = np.arange(2000*6.).reshape(2000, 3, 2)[:, ::-1]
    copied = ht.parallel_copy(source)
    np.testing.assert_array_equal(copied, source)
    assert copied.flags.c_contiguous and not np.shares_memory(copied, source)


def test_threaded_face_tables_and_point_maps_match_serial():
    space = DGSpace(rectangle_mesh(20, 20), 2, basis_type="dub_orth")
    trace = space.trace_space("legacy-lagrange")
    assert space.mesh.num_tri >= 2*ht.MIN_CHUNK  # several chunks with 4 threads
    rng = np.random.default_rng(3)
    normal = rng.normal(size=(space.mesh.num_tri, 3, trace.weights.size))
    tau = np.abs(normal) + .5
    points = space.quad_data.Krf_quads
    builders = (
        lambda: mats.boundary_mass_from_trace_stabilization(space, tau, trace_space=trace),
        lambda: mats.element_boundary_mats_from_trace_weight(space, tau - normal, trace_space=trace),
        lambda: mats.advection_trace_lift_from_stabilization(space, tau, trace_space=trace),
        lambda: mats.advection_interior_trace_mass_blocks_from_weight(space, tau - normal, trace_space=trace),
        lambda: space.mesh.map_reference_points(points),
    )
    previous = ht.set_host_threads(1)
    try:
        serial = [build() for build in builders]
        ht.set_host_threads(4)
        threaded = [build() for build in builders]
    finally:
        ht.set_host_threads(previous)
    for expected, actual in zip(serial, threaded):
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    mapped = np.einsum("Krc,qc->Kqr", space.mesh.aff_mats, points) + space.mesh.aff_vecs[:, None, :]
    np.testing.assert_allclose(threaded[-1], mapped, rtol=1e-14, atol=1e-15)


def test_manufactured_case_evaluations_are_chunked_transparently(four_threads):
    from scripts.n_gamma.cases import forcing, get_case

    case = get_case("transient_baseline", geometry="cartesian")
    x, y = np.random.default_rng(4).uniform(-1., 1., (2, 800, 100))
    np.testing.assert_array_equal(case.density_source(x, y, .3),
                                  forcing.S_n(x, y, .3, geometry="cartesian"))
    np.testing.assert_array_equal(case.momentum_source(x, y, .3),
                                  forcing.S_Gamma(x, y, .3, geometry="cartesian"))
    b_1, b_2 = case.b_poloidal(x, y)
    np.testing.assert_array_equal(np.stack((b_1, b_2), -1), forcing.b_p(x, y))
    assert case.density(.1, .2, .3) == forcing.n_e(.1, .2, .3)


def test_chunked_diffusion_tables_match_whole_array_arithmetic(four_threads):
    from hdgfem.assembly import diffusion_coefficients as dc

    rng = np.random.default_rng(2)
    a, d = 1. + rng.random((700, 9)), 1. + rng.random((700, 9))
    off = .3*rng.random((700, 9))
    values = np.stack((a, off, off + .01*(rng.random((700, 9)) < .5), d), axis=-1)
    values[:100] = values[:100, :1]         # constant elements
    dc.validate_diffusion_values(values)
    np.testing.assert_array_equal(dc.inverse_diffusion_values(values), dc._inverse(values, np))
    kinds = dc.diffusion_kinds(values)
    np.testing.assert_array_equal(kinds, dc._diffusion_kinds(values, np))
    assert set(kinds[:100]) == {2} and set(kinds[100:]) <= {5, 6}
    broken = values.copy()
    broken[5, 0, 0], broken[650, 3, 3] = -1., np.nan   # the nonfinite sample is reported first
    with pytest.raises(ValueError, match='finite'):
        dc.validate_diffusion_values(broken)
    broken[650, 3, 3] = 1.
    with pytest.raises(ValueError, match='positive definite'):
        dc.validate_diffusion_values(broken)
