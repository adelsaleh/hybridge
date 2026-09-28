"""Small coefficient checks; compilation/GPU execution requires explicit opt-in."""
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.cases import closed_loop_stress_cases as reference
from scripts.advection_diffusion_reaction.cases.closed_loop_stress_sampling import StressCoefficientSampler


ROOT = Path(__file__).resolve().parents[1]


def adapter(variant="trap", backend="numpy", **options):
    parameters = reference.StressParameters(variant=variant)
    spec = dict(master_root=str(ROOT), stress_parameters=parameters.to_dict(),
                velocity_normalization=1.0, coefficient_backend=backend,
                coefficient_chunk_points=9, **options)
    return StressCoefficientSampler(spec), parameters


def points(parameters):
    rho = np.linspace(0.03, 0.97, 5)[None, :]
    phi = ((2*np.arange(9)+1)*np.pi/9)[:, None]
    return reference.polar_points(rho, phi, parameters.hole_radius)


@pytest.mark.parametrize("variant", reference.VARIANTS)
def test_bundled_matches_independent_reference(variant):
    evaluator, p = adapter(variant)
    x, y = points(p)
    kwargs, _ = reference.make_case(p, 1.0)
    expected = np.array([kwargs["diffusion"][0][0](x, y), kwargs["diffusion"][0][1](x, y),
                         kwargs["diffusion"][1][1](x, y), kwargs["beta"][0](x, y),
                         kwargs["beta"][1](x, y), kwargs["source"](x, y)])
    result = evaluator.volume(x, y)
    np.testing.assert_allclose(result, expected, rtol=5e-12, atol=2e-8)
    np.testing.assert_allclose(evaluator.velocity(x, y), expected[3:5], rtol=5e-13, atol=1e-10)
    assert evaluator.stats["cpu_batches"] == 18  # one complete element per batch
    assert evaluator.stats["sampled_points"] == 2*x.size


def simple(x, y, parameters):
    return x+y, parameters[0]*x-y


def test_scalar_broadcast_empty_and_strided_inputs():
    sampler = adapter()[0].sampler
    for x, y in [(2.0, 3.0), (np.arange(12.).reshape(3, 4)[:, ::2], 2.0),
                 (np.empty((0, 5)), np.empty((0, 5)))]:
        x, y = np.broadcast_arrays(x, y)
        actual = sampler.sample(simple, x, y, (3.,), components=2)
        np.testing.assert_array_equal(actual, np.array(simple(x, y, (3.,))))


def test_scalar_kernel_body_without_compiling():
    evaluator, p = adapter()
    import sys
    module = sys.modules["_hdgfem_coefficient_sampling"]
    x, y = points(p)
    result = np.empty((6, *x.shape))
    module._point_loop(evaluator.formulas.closed_loop_volume, evaluator.parameters, x, y, result)
    np.testing.assert_allclose(result, evaluator.volume(x, y), rtol=5e-12, atol=2e-8)


class FakeOOM(Exception):
    pass


class FakePool:
    malloc = None
    def free_bytes(self):
        return 0
    def free_all_blocks(self):
        pass


def fake_gpu(sampler, free_bytes=10**10):
    cp = SimpleNamespace(
        asarray=np.asarray, broadcast_to=np.broadcast_to, asnumpy=np.asarray, float64=np.float64,
        cuda=SimpleNamespace(Device=lambda device: nullcontext(), using_allocator=lambda allocator: nullcontext(),
                             runtime=SimpleNamespace(memGetInfo=lambda: (free_bytes, free_bytes)),
                             memory=SimpleNamespace(OutOfMemoryError=FakeOOM),
                             get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    sampler._gpu_checked, sampler._cp, sampler._pool = True, cp, FakePool()
    return cp


def test_gpu_oom_shrinks_and_preserves_all_samples():
    sampler = adapter(backend="auto")[0].sampler
    fake_gpu(sampler)
    sizes = []
    def function(x, y, parameters):
        sizes.append(x.size)
        if x.size > 2:
            raise FakeOOM()
        return simple(x, y, parameters)
    x, y = np.arange(7.), np.arange(7.)+2
    result = sampler.sample(function, x, y, (3.,), components=2)
    np.testing.assert_array_equal(result, np.array(simple(x, y, (3.,))))
    assert sampler.stats["oom_retries"] == 2
    assert sampler.stats["gpu_batches"] == 7
    assert sizes[:3] == [7, 3, 1]


def test_gpu_memory_reserve_uses_cpu_without_compiling(monkeypatch):
    sampler = adapter(backend="auto")[0].sampler
    fake_gpu(sampler, free_bytes=1)
    def cpu(function, parameters, x, y, out):
        out[:] = function(x, y, parameters)
        sampler.stats["cpu_batches"] += 1
    monkeypatch.setattr(sampler, "_cpu", cpu)
    result = sampler.sample(simple, [1., 2.], [3., 4.], (3.,), components=2)
    np.testing.assert_array_equal(result, [[4., 6.], [0., 2.]])
    assert sampler.stats["fallback_reason"] == "GPU sampling memory reserve"
    assert sampler.stats["gpu_batches"] == 0


def test_gpu_programming_errors_are_not_hidden():
    sampler = adapter(backend="auto")[0].sampler
    fake_gpu(sampler)
    def bad(x, y, parameters):
        raise ValueError("coefficient bug")
    with pytest.raises(ValueError, match="coefficient bug"):
        sampler.sample(bad, [1.], [2.], components=1)
    assert sampler.stats["cpu_batches"] == 0


def test_invalid_and_nonfinite():
    sampler = adapter()[0].sampler
    with pytest.raises(ValueError, match="component count"):
        sampler.sample(simple, [1.], [2.], (3.,), components=1)
    with pytest.raises(ValueError, match="Nonfinite"):
        sampler.sample(simple, [np.nan], [2.], (3.,), components=2)


@pytest.mark.skipif(os.environ.get("HDGFEM_TEST_SAMPLING_JIT") != "1", reason="requires explicit JIT authorization")
@pytest.mark.parametrize("variant", reference.VARIANTS)
def test_compiled_numba_parity(variant):
    evaluator, p = adapter(variant, backend="numba")
    expected, _ = adapter(variant)
    x, y = points(p)
    np.testing.assert_allclose(evaluator.volume(x, y), expected.volume(x, y), rtol=5e-12, atol=2e-8)
    np.testing.assert_allclose(evaluator.velocity(x, y), expected.velocity(x, y), rtol=5e-12, atol=2e-8)
    assert evaluator.stats["numba_threads"] > 0


@pytest.mark.skipif(os.environ.get("HDGFEM_TEST_SAMPLING_CUDA") != "1", reason="requires explicit CUDA/JIT authorization")
@pytest.mark.parametrize("variant", reference.VARIANTS)
def test_cupy_parity(variant):
    evaluator, p = adapter(variant, backend="cupy")
    expected, _ = adapter(variant)
    x, y = points(p)
    np.testing.assert_allclose(evaluator.volume(x, y), expected.volume(x, y), rtol=5e-11, atol=2e-7)
    np.testing.assert_allclose(evaluator.velocity(x, y), expected.velocity(x, y), rtol=5e-11, atol=2e-7)
    assert evaluator.stats["gpu_batches"] > 0
