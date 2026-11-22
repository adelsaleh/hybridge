"""Split3 stage budgets, isolation, persistence and numerical qualification."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.runtime import cuda_tuning as tuning
from hybridge.transport import tsle_tuning as stages


@pytest.mark.parametrize("count", range(1, 13))
def test_stages_share_one_budget_including_conservative_candidates(count):
    budgets = stages.stage_budgets(tuning.TuningBudget(30, count))
    assert sum(b.candidates for b in budgets if b is not None) == count
    assert sum(b.seconds for b in budgets if b is not None) <= 30
    if count == 8:
        assert [b.candidates for b in budgets] == [3, 3, 2]


def test_persistent_stage_cache_hit_skips_all_trial_preparation(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING", "auto")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CANDIDATES", "8")
    monkeypatch.setattr(tuning, "device_signature", lambda cp: dict(
        fp32_fp64_ratio=2, maxThreadsPerBlock=1024, sharedMemPerBlock=100000, regsPerBlock=65536))
    attributes = {"num_regs": 16, "max_threads_per_block": 1024}
    kernels = SimpleNamespace(**{stage: SimpleNamespace(attributes=attributes) for stage in stages.STAGES})
    trials, calls = [], []

    class Trial:
        num_elements = 12

        def before(self, stage, baseline):
            pass

        def launch(self, stage, block):
            calls.append((stage, block))
            return np.array([np.nan if block == 128 else 1.])

        def validate(self, stage, output):
            return True

    def factory():
        trials.append(Trial())
        return trials[-1]

    def event(cp, call):
        call()
        return 1. / calls[-1][1]

    monkeypatch.setattr(tuning, "event_seconds", event)
    tuning.clear_tuning_cache()
    kwargs = dict(cupy=np, identity=dict(source="test-source", precision="float64", stabilization=1),
        kernels=kernels, shared_bytes={stage: lambda block: 1000 for stage in stages.STAGES},
        make_trial=factory, nel=55)
    first = {}
    selected = stages.select_stage_blocks(**kwargs, timings=first)
    assert selected == (256, 64, 64)
    assert len(trials) == 1 and first["raw.tsle.autotune.candidates"] == 8
    assert len(list(tmp_path.glob("*.json"))) == 3
    tuning.clear_tuning_cache()
    kwargs["make_trial"] = lambda: pytest.fail("disk hits must not allocate calibration scratch")
    previous_calls = len(calls)
    second = {}
    assert stages.select_stage_blocks(**kwargs, timings=second) == selected
    assert second["raw.tsle.autotune.reused"] == 1
    assert second["raw.tsle.autotune.sample_elements"] == 0
    assert len(calls) == previous_calls
    assert "raw.tsle.autotune.wall" not in second


def test_compiler_resources_are_checked_before_scratch(monkeypatch):
    monkeypatch.setattr(tuning, "device_signature", lambda cp: dict(regsPerBlock=65536))
    kernel = SimpleNamespace(attributes={"num_regs": 4096})
    with pytest.raises(tuning.CandidateUnavailable, match="compiled/device resources"):
        stages.select_stage_blocks(cupy=np, identity={},
            kernels=SimpleNamespace(build=kernel, solve=kernel, scatter=kernel),
            shared_bytes={stage: lambda block: 100 for stage in stages.STAGES},
            make_trial=lambda: pytest.fail("illegal candidates must not allocate scratch"), nel=3, timings={})


def test_shared_deadline_prevents_later_stage_calibration_after_overrun(tmp_path, monkeypatch):
    clock = [100.]
    fake_time = SimpleNamespace(perf_counter=lambda: clock[0])
    monkeypatch.setattr(stages, "time", fake_time)
    monkeypatch.setattr(tuning, "time", fake_time)
    monkeypatch.setattr(tuning, "device_signature", lambda cp: {})
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING", "refresh")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_SECONDS", "30")
    kernel = SimpleNamespace(attributes={"num_regs": 16})
    launches = []

    class Trial:
        num_elements = 1

        def before(self, stage, baseline):
            clock[0] += 31.

        def launch(self, stage, block):
            launches.append(stage)
            return np.ones(1)

        def validate(self, stage, output):
            return True

    selected = stages.select_stage_blocks(cupy=np, identity={"overrun": True},
        kernels=SimpleNamespace(build=kernel, solve=kernel, scatter=kernel),
        shared_bytes={stage: lambda block: 100 for stage in stages.STAGES},
        make_trial=Trial, nel=3, timings={})
    assert selected == (32, 32, 32)
    assert launches == ["build"]


def test_explicit_blocks_do_not_query_device_or_prepare_trials():
    from hybridge.transport.tsle_bsr import _select_launch_blocks
    kwargs = dict(cupy=None, identity={}, kernels=None, nel=3, ntr=2, ncols=7, nqf=3,
                  timings={}, make_trial=lambda: pytest.fail("explicit tuning"))
    assert _select_launch_blocks(**kwargs, requested=256) == (256, 256, 256)
    with pytest.raises(ValueError, match="block size"):
        _select_launch_blocks(**kwargs, requested=16)


def test_tuning_identity_uses_real_space_reference_basis_and_quadrature():
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.transport.tsle_bsr import _tuning_space_identity

    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8, edge_quad_1d=5)
    assert _tuning_space_identity(space) == dict(order=3, basis="dub_orth",
        quadrature=dict(volume="duffy", volume_1d=8, volume_degree=None, edge_1d=5))
    changed = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=9, edge_quad_1d=5)
    assert _tuning_space_identity(changed) != _tuning_space_identity(space)


def test_unavailable_trial_memory_uses_baselines_without_persisting_them(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING", "auto")
    monkeypatch.setattr(tuning, "device_signature", lambda cp: {})
    kernel = SimpleNamespace(attributes={"num_regs": 16})
    attempts = []

    def unavailable():
        attempts.append(True)
        raise stages.TuningScratchUnavailable("no scratch")

    timings = {}
    assert stages.select_stage_blocks(cupy=np, identity={"memory-test": True},
        kernels=SimpleNamespace(build=kernel, solve=kernel, scatter=kernel),
        shared_bytes={stage: lambda block: 100 for stage in stages.STAGES},
        make_trial=unavailable, nel=3, timings=timings) == (32, 32, 32)
    assert len(attempts) == 1
    assert timings["raw.tsle.autotune.scratch_unavailable"] == 1.
    assert not list(tmp_path.glob("*.json"))


def test_solve_trials_restore_inputs_and_check_independent_backward_error():
    operator = np.array([[[2., 0.], [0., 4.]]])
    response = np.full((1, 2, 2), 8.)

    def build(*args, **kwargs):
        response.fill(8.)

    def solve(*args, **kwargs):
        response[:] = np.linalg.solve(operator, response)

    trial = stages.StageTrial(np, SimpleNamespace(build=build, solve=solve),
        dict(build=(operator, response, np.ones((1, 2, 3, 2))), solve=(operator, response, 1)),
        dict(build=lambda block: 0, solve=lambda block: 0), 1, "nodal", False)
    baseline = dict(build=32, solve=32)
    for _ in range(3):
        trial.before("solve", baseline)
        trial.launch("solve", 64)
        assert trial.validate("solve", response)
        np.testing.assert_array_equal(response, [[[4., 4.], [2., 2.]]])
    response[0, 0, 0] += 1
    assert not trial.validate("solve", response)
    response[0, 0, 0] = np.nan
    assert not trial.validate("solve", response)


def _scatter_fixture():
    from hybridge import rectangle_mesh
    mesh = rectangle_mesh(1, 1)
    edge = int(mesh.int_edges_inds[0])
    solve_edges = np.full(mesh.num_edg, -1, dtype=np.int64)
    solve_edges[edge] = 0
    side_index = np.full((2, 3), -1, dtype=np.int64)
    side_index[mesh.loc2glob_edge == edge] = np.flatnonzero(mesh.loc2glob_edge.ravel() == edge)
    sides = np.full((mesh.num_edg, 2), -1, dtype=np.int64)
    for element in range(2):
        for face in range(3):
            current = mesh.loc2glob_edge[element, face]
            sides[current, 0 if sides[current, 0] < 0 else 1] = element*3 + face
    response = np.zeros((2, 2, 7))
    for face in range(3):
        response[:, :, 2*face:2*face+2] = np.eye(2)
    response[:, :, -1] = 1.
    return (np.array([0, 1], dtype=np.int32), np.zeros((1, 2, 2)), np.zeros(2), response,
        np.ones((2, 2, 3, 2)), mesh.loc2glob_edge, sides, mesh.orientations,
        side_index.ravel(), solve_edges, np.zeros(18, dtype=np.int32), np.array([0], dtype=np.int32),
        np.tile(np.eye(2), (3, 1, 1)), np.eye(2), np.ones(mesh.num_edg*2), 2)


@pytest.mark.parametrize("orientation,expected", [("nodal", [6., 6.]), ("modal", [6., 0.])])
def test_independent_scatter_oracle_orientation_boundary_and_duplicate_accumulation(orientation, expected):
    arguments = _scatter_fixture()
    data, rhs = stages.scatter_oracle(arguments, orientation=orientation, conflict_averaged=False)
    np.testing.assert_array_equal(data, 0.)
    np.testing.assert_array_equal(rhs, expected)
    arguments[4].fill(0.)
    data, rhs = stages.scatter_oracle(arguments, orientation=orientation, conflict_averaged=True)
    np.testing.assert_array_equal(data[0], np.eye(2))
    np.testing.assert_array_equal(rhs, 0.)


def _dense_scatter_fixture(dtype):
    """Exercise every scatter contribution with a small overcomplete BSR map."""
    from hybridge import rectangle_mesh
    from hybridge.hdg.trace_maps import _edge_to_solve_edge, _interior_side_index
    mesh = rectangle_mesh(2, 2)
    ntr, nel, nqf = 3, 4, 5
    rows = mesh.int_edges_inds.size
    solve_edges = _edge_to_solve_edge(mesh)
    random = np.random.default_rng(981)

    def values(shape):
        """Retain the requested precision in independent random test inputs."""
        return random.normal(size=shape).astype(dtype)

    flux = values((mesh.num_tri, 2, 3, nqf))
    # The zero-flux interior face must receive exactly one optional gauge row.
    gauge = mesh.edge_side_indices[mesh.int_edges_inds[0]]
    flux[gauge // 3, :, gauge % 3] = 0
    return (np.arange(rows + 1, dtype=np.int32) * rows,
        np.zeros((rows*rows, ntr, ntr), dtype=dtype), np.zeros(rows*ntr, dtype=dtype),
        values((mesh.num_tri, nel, 3*ntr+1)), flux, mesh.loc2glob_edge,
        mesh.edge_side_indices, mesh.orientations, _interior_side_index(mesh), solve_edges,
        solve_edges[mesh.loc2glob_edge[mesh.interior_elements]].astype(np.int32),
        np.arange(rows, dtype=np.int32), values((3, nel, nqf)), values((ntr, nqf)),
        values((mesh.num_edg, ntr)), mesh.num_tri)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("orientation", ["nodal", "modal"])
@pytest.mark.parametrize("conflict_averaged", [False, True])
def test_device_scatter_algebra_matches_serial_oracle_and_rejects_bad_outputs(
        dtype, orientation, conflict_averaged):
    """Cover off-diagonal blocks, signs, gauge rows and incorrect candidates."""
    arguments = _dense_scatter_fixture(dtype)
    options = dict(orientation=orientation, conflict_averaged=conflict_averaged)
    expected = stages.scatter_oracle(arguments, **options)
    actual = stages.device_scatter_oracle(arguments, array_module=np, **options)
    for value, reference in zip(actual, expected):
        assert value.dtype == np.dtype(dtype)
        np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-6)
    trial = stages.StageTrial(np, None, dict(scatter=arguments, solve=(arguments[3],)), {},
                              int(arguments[-1]), orientation, conflict_averaged)
    assert trial.validate("scatter", actual)
    actual[0][0, 0, 0] += 1.
    assert not trial.validate("scatter", actual)
    actual[0][0, 0, 0] = np.nan
    assert not trial.validate("scatter", actual)


def _gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("orientation", ["nodal", "modal"])
def test_gpu_scatter_validation_keeps_oracle_and_outputs_on_device(dtype, orientation, monkeypatch):
    """Forbid array downloads even on the first correctness-checked trial."""
    import cupy as cp
    host = _dense_scatter_fixture(dtype)
    expected = stages.scatter_oracle(host, orientation=orientation, conflict_averaged=True)
    arguments = tuple(cp.asarray(value) if isinstance(value, np.ndarray) else value for value in host)
    output = tuple(cp.asarray(value) for value in expected)
    trial = stages.StageTrial(cp, None, dict(scatter=arguments, solve=(arguments[3],)), {},
                              int(arguments[-1]), orientation, True)
    monkeypatch.setattr(cp, "asnumpy", lambda *args, **kwargs: pytest.fail("calibration downloaded an array"))
    assert trial.scatter_reference is None
    assert trial.validate("scatter", output)
    assert all(isinstance(value, cp.ndarray) for value in trial.scatter_reference)
    output[1][0] += 1.
    assert not trial.validate("scatter", output)


@pytest.mark.skipif(not _gpu_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("order,trace_basis,zero_flux,stabilization", [
    (3, "legacy-lagrange", False, None),
    (6, "legendre-modal", True, "conflict-averaged-upwind"),
    (9, "legendre-modal", False, None),
])
@pytest.mark.parametrize("volume", ["inline", "cublas"])
def test_gpu_bounded_stage_tuning_matches_explicit_and_disk_hit_skips_sampling(
        order, trace_basis, zero_flux, stabilization, volume, tmp_path, monkeypatch):
    import json
    import cupy as cp
    from test_advection_tsle_tensor import _split3_inputs
    from hybridge.runtime.precision import REAL_ITEMSIZE
    from hybridge.transport import tsle_bsr, tuning_sample
    if volume != "inline" and REAL_ITEMSIZE != 8:
        pytest.skip("the optional preloaded volume prototype supports FP64 only")
    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", volume)
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING", "auto")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CANDIDATES", "8")
    monkeypatch.setattr(tsle_bsr, "_TSLE_TUNE_LIMIT", 5)
    inputs = _split3_inputs(order, trace_basis, zero_boundary_flux=zero_flux, stabilization=stabilization)
    expected = tsle_bsr.assemble_projected_advection_trace_system_eliminated_tsle_bsr(**inputs)
    inputs["block_size"] = "auto"
    tuning.clear_tuning_cache()
    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", lambda *args, **kwargs: pytest.fail("cold calibration downloaded an array"))
        actual = tsle_bsr.assemble_projected_advection_trace_system_eliminated_tsle_bsr(**inputs)
    assert actual.timings["raw.tsle.autotune.sample_elements"] == 5
    tolerance = 2.e-12 if REAL_ITEMSIZE == 8 else 4.e-5
    for name in ("data", "rhs", "local_response"):
        cp.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=tolerance, atol=tolerance)
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert {record["workload"]["stage"] for record in records} == set(stages.STAGES)
    assert sum(len(record["candidates"]) for record in records) <= 8
    tuning.clear_tuning_cache()
    monkeypatch.setattr(tuning_sample, "sample_raw_options",
                        lambda *args, **kwargs: pytest.fail("production disk hit prepared trial mesh"))
    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", lambda *args, **kwargs: pytest.fail("cached calibration downloaded an array"))
        cached = tsle_bsr.assemble_projected_advection_trace_system_eliminated_tsle_bsr(**inputs)
    assert cached.timings["raw.tsle.autotune.reused"] == 1
    for name in ("data", "rhs", "local_response"):
        cp.testing.assert_allclose(getattr(cached, name), getattr(expected, name), rtol=tolerance, atol=tolerance)
