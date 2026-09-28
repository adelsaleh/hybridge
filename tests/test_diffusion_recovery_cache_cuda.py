"""Small prescribed-field CUDA recovery checks; no PDE solve or time stepping."""

import numpy as np
import pytest

from hdgfem.backends import diffusion_flux_recovery_raw_cuda as raw
from hdgfem.core.mesh import DGMesh, rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers import diffusion_reaction as diffusion


@pytest.fixture
def cp():
    module = pytest.importorskip("cupy")
    try:
        count = module.cuda.runtime.getDeviceCount()
    except module.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA is unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA device")
    return module


@pytest.mark.parametrize("degree", [0, 2, 5])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("variant", ["RT_projection", "l2_closest"])
def test_solver_tau_retries_reuse_device_recovery_without_uploads_or_refactorization(
        cp, monkeypatch, degree, trace_basis, variant):
    mesh = rectangle_mesh(1, 1)
    mesh = DGMesh.from_arrays(
        mesh.node_coords @ np.array([[1.8, .4], [-.2, .7]]) + [.2, -.1], mesh.triangles)
    space = DGSpace(mesh, degree, basis_type="dub_orth")
    trace_space = space.trace_space(trace_basis)
    options = diffusion.DiffusionReactionHDGOptions(
        stabilization=.4, assembly_backend="raw-cuda", boundary_mode="eliminate",
        trace_basis=trace_basis, hdg_postprocess="flux", flux_postprocess_space=variant,
        postprocessing_backend="auto", verbose=False,
    )
    solver = diffusion.DiffusionReactionHDGSolver(
        space, source=space.zeros(), reaction=space.zeros(), boundary_condition=0., options=options)
    rng = np.random.default_rng(570)

    def prescribed_result():
        local = rng.normal(size=(mesh.num_tri, 3 * space.el_dof))
        trace = rng.normal(size=mesh.num_edg * trace_space.edg_dof)
        result = diffusion.DiffusionReactionResult(
            field=space.zeros(), flux=None, trace=None,
            timings=diffusion.DiffusionReactionTimings(0., 0., 0., 0., 0., 0., 0.),
            local_unknowns_device=cp.asarray(local), trace_device=cp.asarray(trace),
        )
        return result, local, trace

    first, _, _ = prescribed_result()
    solver._postprocess_result(first)
    original = solver._hdg_postprocess_cache
    device_cache = original.raw_flux_cache
    pointers = tuple(array.data.ptr for array in device_cache.arrays) + (device_cache.cholesky.data.ptr,)
    original_asarray = cp.asarray

    def forbid_rebuild(*args, **kwargs):
        raise AssertionError("Tau-only retry rebuilt recovery reference data or geometry factors")

    def forbid_download(*args, **kwargs):
        raise AssertionError("Tau-only retry downloaded a full field or trace")

    def device_arrays_only(value, *args, **kwargs):
        if not isinstance(value, cp.ndarray):
            raise AssertionError("Tau-only retry uploaded host recovery data")
        return original_asarray(value, *args, **kwargs)

    for tau in (.8, 2.3):
        result, local, trace = prescribed_result()
        expected = diffusion._postprocess_diffusion_solution(
            local, trace, space, tau, 1., "flux", trace_space=trace_space,
            flux_postprocess_space=variant, postprocessing_backend="numba",
        )[1]
        fresh, fresh_cache = raw.recover_diffusion_flux_raw_cuda(
            result.local_unknowns_device, result.trace_device, space, trace_space, tau, variant)
        assert fresh_cache is not device_cache
        with monkeypatch.context() as guard:
            guard.setattr(diffusion, "_new_hdg_postprocess_cache", forbid_rebuild)
            guard.setattr(raw, "build_flux_recovery_reference", forbid_rebuild)
            guard.setattr(raw, "as_cupy_space", forbid_rebuild)
            guard.setattr(cp, "asnumpy", forbid_download)
            guard.setattr(cp, "asarray", device_arrays_only)
            solver.with_options(stabilization=tau)
            solver.set_source(space.constant(tau))
            solver.set_boundary_condition(tau)
            recovered = solver._postprocess_result(result).postprocessed_flux
            assert all(not component.coefficients_materialized for component in recovered.components)
            retained = solver._hdg_postprocess_cache
            assert retained.raw_flux_cache is device_cache
            assert retained.post_space is original.post_space
            assert retained.base_to_post_mass is original.base_to_post_mass
            actual_pointers = tuple(array.data.ptr for array in device_cache.arrays) + (device_cache.cholesky.data.ptr,)
            assert actual_pointers == pointers
        np.testing.assert_allclose(
            recovered.as_component_first(), fresh.as_component_first(), rtol=2.e-9, atol=2.e-9)
        np.testing.assert_allclose(
            recovered.as_component_first(), expected.as_component_first(), rtol=2.e-9, atol=2.e-9)
