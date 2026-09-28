"""Inject AMGX allocation failures without a GPU, native build, or large matrix."""

from collections import Counter
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from hdgfem.backends import advection_cuda as raw_amgx
from hdgfem.backends import cupy as cupy_backend
from hdgfem.backends.amgx_errors import as_amgx_capacity_error, is_amgx_capacity_error
from hdgfem.linalg.system import LinearSolveCapacityError


class AMGXAllocationError(RuntimeError):
    error_code = 7


class DeviceArray(np.ndarray):
    """Small NumPy stand-in exposing the pointer used by AMGX vector uploads."""

    def __new__(cls, values, runtime, *, matrix=False, dtype=None):
        result = np.asarray(values, dtype=dtype).view(cls)
        result.runtime = runtime
        result.is_matrix = matrix
        return result

    def __array_finalize__(self, source):
        self.runtime = getattr(source, "runtime", None)
        self.is_matrix = getattr(source, "is_matrix", False)

    @property
    def data(self):
        return SimpleNamespace(ptr=self.ctypes.data)

    def copy(self, *args, **kwargs):
        if self.is_matrix:
            raise AssertionError("AMGX retries must not copy the full device matrix")
        self.runtime.trip("solution.copy")
        return super().copy(*args, **kwargs)


class NativeObject:
    """Record ownership and raise at a selected native API boundary."""

    status = "SUCCESS"
    iterations_number = 0

    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name
        self.destroyed = False

    def operation(self, name):
        self.runtime.last_operation = f"{self.name}.{name}"
        self.runtime.trip(self.runtime.last_operation)
        return self

    def create_from_dict(self, *_args, **_kwargs):
        return self.operation("create")

    create = create_from_dict
    create_simple = create_from_dict

    def upload(self, *_args, **_kwargs):
        return self.operation("upload")

    upload_CSR = upload
    upload_raw = upload

    def setup(self, *_args, **_kwargs):
        return self.operation("setup")

    def solve(self, *_args, **_kwargs):
        return self.operation("solve")

    def download_raw(self, *_args, **_kwargs):
        return self.operation("download")

    def replace_coefficients(self, *_args, **_kwargs):
        return self.operation("replace")

    def get_nnz(self):
        return 2

    def destroy(self):
        assert not self.destroyed, f"double destroy: {self.name}"
        self.destroyed = True
        self.runtime.events.append(f"{self.name}.destroy")
        if self.name in self.runtime.destroy_failures:
            raise RuntimeError(f"cannot destroy {self.name}")


class FakeRuntime:
    """In-memory CUDA/AMGX API double with measurable native object lifetimes."""

    def __init__(self):
        self.fail_at = None
        self.last_operation = "idle"
        self.events = []
        self.objects = []
        self.counts = Counter()
        self.destroy_failures = set()
        self.cp = SimpleNamespace(
            int32=np.int32,
            asarray=self.asarray,
            zeros_like=self.zeros_like,
            isfinite=np.isfinite,
            all=lambda value: SimpleNamespace(get=lambda: bool(np.all(value))),
            asnumpy=self.asnumpy,
            cuda=SimpleNamespace(
                get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize),
                runtime=SimpleNamespace(memGetInfo=self.device_memory),
            ),
        )
        self.amgx = SimpleNamespace(
            Config=lambda: self.object("config"),
            Resources=lambda: self.object("resources"),
            Matrix=lambda: self.object("matrix"),
            Vector=lambda: self.object("vector"),
            Solver=lambda: self.object("solver"),
            get_device_memory_stats=self.amgx_memory,
        )

    def trip(self, event):
        self.events.append(event)
        if event == self.fail_at:
            raise AMGXAllocationError("CUDA allocation failed")

    def object(self, kind):
        self.counts[kind] += 1
        obj = NativeObject(self, f"{kind}-{self.counts[kind]}")
        self.objects.append(obj)
        return obj

    def asarray(self, values, dtype=None):
        if isinstance(values, DeviceArray) and (dtype is None or values.dtype == dtype):
            return values
        return DeviceArray(values, self, dtype=dtype)

    def zeros_like(self, value):
        self.trip("solution.allocate")
        return DeviceArray(np.zeros_like(np.asarray(value)), self)

    def asnumpy(self, _value):
        raise AssertionError("capacity handling must not download device arrays")

    def synchronize(self):
        self.trip(f"{self.last_operation}.sync")

    def amgx_memory(self):
        self.events.append("memory.amgx")
        return {
            "live_bytes": sum(not obj.destroyed for obj in self.objects) * 1024,
            "reserved_bytes": 32768,
            "peak_live_bytes": 16384,
            "peak_reserved_bytes": 65536,
        }

    def device_memory(self):
        self.events.append("memory.device")
        return 2 * 1024**3, 8 * 1024**3

    def assembly(self, block_size=1):
        data = [1.0, 1.0] if block_size == 1 else [[[1.0, 0.0], [0.0, 1.0]]]
        return SimpleNamespace(
            matrix_format="csr" if block_size == 1 else "bsr",
            data=DeviceArray(data, self, matrix=True),
            indices=np.array([0, 1] if block_size == 1 else [0], dtype=np.int32),
            indptr=np.array([0, 1, 2] if block_size == 1 else [0, 1], dtype=np.int32),
            rhs=DeviceArray([1.0, 2.0], self),
            shape=(2, 2),
            block_size=block_size,
        )


@pytest.fixture
def runtime(monkeypatch):
    runtime = FakeRuntime()
    for module in (raw_amgx, cupy_backend):
        monkeypatch.setattr(module, "require_cupy", lambda: runtime.cp)
        monkeypatch.setattr(module, "initialize_pyamgx_once", lambda: runtime.amgx)
    monkeypatch.setattr(raw_amgx, "require_pyamgx", lambda: runtime.amgx)
    monkeypatch.setattr(raw_amgx, "require_cupyx_sparse", lambda: None)
    monkeypatch.setattr(raw_amgx, "audit_arrays", lambda *_args: None)
    monkeypatch.setattr(raw_amgx, "_AMGX_SHARED_RESOURCES", raw_amgx._PyAMGXSharedResourceManager())
    monkeypatch.setattr(raw_amgx, "_AMGX_REUSABLE_SOLVERS", [])
    yield runtime
    for solver in raw_amgx._AMGX_REUSABLE_SOLVERS:
        solver.close(suppress_errors=True)


RAW_FAILURES = (
    ("config-1.create", "resource acquisition"),
    ("resources-1.create", "resource acquisition"),
    ("config-2.create", "configuration creation"),
    ("matrix-1.create", "solver-object creation"),
    ("vector-1.create", "solver-object creation"),
    ("vector-2.create", "solver-object creation"),
    ("solver-1.create", "solver-object creation"),
    ("matrix-1.upload", "matrix upload"),
    ("matrix-1.upload.sync", "matrix upload"),
    ("solver-1.setup", "solver setup"),
    ("solver-1.setup.sync", "setup synchronization"),
    ("solution.allocate", "solution allocation"),
    ("vector-1.upload", "vector upload"),
    ("vector-2.upload", "vector upload"),
    ("solver-1.solve", "solver iteration"),
    ("vector-2.download", "solution download"),
    ("vector-2.download.sync", "solution download"),
)


def assert_released(runtime):
    assert runtime.objects
    assert all(obj.destroyed for obj in runtime.objects)
    manager = raw_amgx._AMGX_SHARED_RESOURCES
    assert manager.refcount == 0
    assert manager.rsrc is manager.resource_cfg is manager.pyamgx is None


@pytest.mark.parametrize("event,phase", RAW_FAILURES)
@pytest.mark.parametrize("block_size", (1, 2), ids=("csr", "bsr"))
@pytest.mark.parametrize("raise_on_nonconvergence", (False, True))
def test_raw_oom_is_terminal_at_every_native_phase(
    runtime, event, phase, block_size, raise_on_nonconvergence
):
    runtime.fail_at = event
    assembly = runtime.assembly(block_size)
    original = np.array(assembly.data)
    with pytest.raises(LinearSolveCapacityError) as raised:
        raw_amgx.solve_reduced_system_amgx_device(
            assembly,
            retry_attempts=({"label": "amgx-retry"}, {"label": "direct-retry", "backend": "cusolver-qr"}),
            scale_system=False,
            materialize_host_solution=False,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )
    error = raised.value
    assert error.phase == phase
    assert error.backend == "pyamgx-device"
    assert error.amgx_attempt_count == 1
    assert len(error.amgx_attempts) == 1
    assert error.amgx_attempts[0]["terminal_capacity_failure"]
    assert isinstance(error.__cause__, AMGXAllocationError)
    assert runtime.events.count(event) == 1
    assert error.memory["amgx"]["live_bytes"] > 0
    assert error.memory["device"]["free_bytes"] == 2 * 1024**3
    first_destroy = next(i for i, entry in enumerate(runtime.events) if entry.endswith(".destroy"))
    assert runtime.events.index("memory.amgx") < first_destroy
    np.testing.assert_array_equal(assembly.data, original)
    assert_released(runtime)


@pytest.mark.parametrize("event,phase", (
    ("config-1.create", "configuration creation"),
    ("resources-1.create", "resource acquisition"),
    *RAW_FAILURES[3:8],
    ("solver-1.setup", "solver setup"),
    *RAW_FAILURES[11:],
))
def test_host_to_amgx_adapter_has_the_same_capacity_and_cleanup_contract(runtime, event, phase):
    runtime.fail_at = event
    matrix = runtime.assembly()
    with pytest.raises(LinearSolveCapacityError) as raised:
        cupy_backend.solve_pyamgx_csr(matrix, matrix.rhs)
    assert raised.value.phase == phase
    assert isinstance(raised.value.__cause__, AMGXAllocationError)
    assert all(obj.destroyed for obj in runtime.objects)
    assert runtime.events.count(event) == 1


@pytest.mark.parametrize("event,phase", (
    ("matrix-1.upload", "matrix upload"),
    ("solver-1.setup", "solver setup"),
    ("matrix-1.replace", "coefficient replacement"),
    ("matrix-1.replace.sync", "coefficient replacement"),
    ("solution.copy", "solution allocation"),
    ("solver-1.solve", "solver iteration"),
))
def test_reusable_solver_closes_after_oom(runtime, event, phase):
    solver = raw_amgx.PyAMGXCsrDeviceSolver(reusable=True)
    matrix = runtime.assembly()
    runtime.fail_at = event
    with pytest.raises(LinearSolveCapacityError) as raised:
        solver.setup(matrix)
        if "replace" in event:
            solver.replace_coefficients(matrix)
        else:
            solver.solve(matrix.rhs, initial_guess=matrix.rhs)
    assert raised.value.phase == phase
    assert solver.closed and not solver.is_setup
    assert_released(runtime)
    solver.close()  # Idempotence includes failed partial setup.


@pytest.mark.parametrize("guess", ([1.0, 2.0, 3.0], [object()]))
def test_invalid_guess_preserves_a_healthy_reusable_solver(runtime, guess):
    solver = raw_amgx.PyAMGXCsrDeviceSolver(reusable=True)
    matrix = runtime.assembly()
    solver.setup(matrix)
    with pytest.raises((ValueError, TypeError)):
        solver.solve(matrix.rhs, initial_guess=guess)
    assert not solver.closed and solver.is_setup
    assert all(not obj.destroyed for obj in runtime.objects)
    solver.close()
    assert_released(runtime)


@pytest.mark.parametrize("adapter", ("raw", "host"))
def test_cleanup_errors_cannot_mask_oom_or_skip_other_owned_objects(runtime, adapter):
    runtime.fail_at = "solver-1.setup"
    runtime.destroy_failures.update({"solver-1", "resources-1"})
    matrix = runtime.assembly()
    with pytest.raises(LinearSolveCapacityError, match="solver setup"):
        if adapter == "raw":
            raw_amgx._pyamgx_solve_csr_device(matrix, matrix.rhs)
        else:
            cupy_backend.solve_pyamgx_csr(matrix, matrix.rhs)
    assert_released(runtime)


def test_shared_resource_failure_resets_manager_for_the_next_call(runtime):
    runtime.fail_at = "resources-1.create"
    runtime.destroy_failures.add("resources-1")
    with pytest.raises(LinearSolveCapacityError, match="resource acquisition"):
        raw_amgx.PyAMGXCsrDeviceSolver()
    assert_released(runtime)
    runtime.fail_at = None
    solver = raw_amgx.PyAMGXCsrDeviceSolver()
    assert raw_amgx._AMGX_SHARED_RESOURCES.refcount == 1
    solver.close()
    assert_released(runtime)


def test_oom_does_not_destroy_another_live_solvers_shared_resources(runtime):
    healthy = raw_amgx.PyAMGXCsrDeviceSolver(reusable=True)
    healthy_objects = tuple(runtime.objects)
    failing = raw_amgx.PyAMGXCsrDeviceSolver(reusable=True)
    runtime.fail_at = "solver-2.setup"
    with pytest.raises(LinearSolveCapacityError):
        failing.setup(runtime.assembly())
    assert raw_amgx._AMGX_SHARED_RESOURCES.refcount == 1
    assert not healthy.closed and failing.closed
    assert all(not obj.destroyed for obj in healthy_objects)
    healthy.close()
    assert_released(runtime)


def test_shared_resource_destroy_failure_still_clears_config_and_manager(runtime):
    solver = raw_amgx.PyAMGXCsrDeviceSolver()
    runtime.destroy_failures.add("resources-1")
    with pytest.raises(RuntimeError, match="cannot destroy resources"):
        solver.close()
    assert solver.closed
    assert_released(runtime)


@pytest.mark.parametrize("block_size,scale_mode", ((1, "left"), (2, "left"), (1, "symmetric")))
@pytest.mark.parametrize("restore_fails", (False, True))
def test_scaled_oom_restores_coefficients_without_a_matrix_backup(
    runtime, monkeypatch, block_size, scale_mode, restore_fails
):
    matrix = runtime.assembly(block_size)
    original = np.array(matrix.data)

    def scale(matrix, rhs):
        matrix.data[...] *= 0.5
        rhs *= 0.5
        return np.full(2, 2.0)

    def restore(matrix, *_args, **_kwargs):
        if restore_fails:
            raise RuntimeError("CUDA restore failed")
        matrix.data[...] *= 2.0

    monkeypatch.setattr(raw_amgx, "_diagonal_scale_csr_rows_in_place", scale)
    monkeypatch.setattr(raw_amgx, "_diagonal_scale_bsr_rows_in_place", scale)
    monkeypatch.setattr(raw_amgx, "symmetric_scale_cupy_csr_in_place", scale)
    monkeypatch.setattr(raw_amgx, "_restore_scaled_csr_rows_in_place", restore)
    monkeypatch.setattr(raw_amgx, "_restore_left_scaled_bsr_rows_in_place", restore)
    runtime.fail_at = "solver-1.setup"
    with pytest.raises(LinearSolveCapacityError) as raised:
        raw_amgx.solve_reduced_system_amgx_device(
            matrix, retry_attempts=({"label": "unused"},),
            scale_system=scale_mode, materialize_host_solution=False,
        )
    assert raised.value.phase == "solver setup"
    assert raised.value.amgx_attempt_count == 1
    assert_released(runtime)
    if restore_fails:
        assert "CUDA restore failed" in raised.value.matrix_restore_error
    else:
        np.testing.assert_array_equal(matrix.data, original)


@pytest.mark.parametrize("cached", (False, True))
def test_oom_in_later_retry_stops_before_the_next_attempt(runtime, monkeypatch, cached):
    matrix = runtime.assembly()
    cache = {}
    if cached:
        solver = raw_amgx.PyAMGXCsrDeviceSolver(reusable=True)
        solver.setup(matrix)
        cache["retry"] = solver
    runtime.fail_at = "matrix-1.replace" if cached else "solver-1.setup"
    native_attempt = raw_amgx._solve_reduced_system_amgx_device_once
    attempts = []

    def attempt(assembly, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("ordinary convergence failure")
        return native_attempt(assembly, **kwargs)

    monkeypatch.setattr(raw_amgx, "_solve_reduced_system_amgx_device_once", attempt)
    with pytest.raises(LinearSolveCapacityError) as raised:
        raw_amgx.solve_reduced_system_amgx_device(
            matrix,
            retry_attempts=(
                {"label": "retry", "reuse_preconditioner": cached},
                {"label": "unused"},
            ),
            retry_solver_cache=cache, scale_system=False, materialize_host_solution=False,
        )
    assert len(attempts) == raised.value.amgx_attempt_count == 2
    assert "terminal_capacity_failure" not in raised.value.amgx_attempts[0]
    assert raised.value.amgx_attempts[1]["terminal_capacity_failure"]
    assert_released(runtime)
    if cached:
        assert cache["retry"].closed


@pytest.mark.parametrize("event", ("matrix-1.upload", "solver-1.setup", "solver-1.solve"))
def test_fixed_operator_cache_does_not_retain_failed_native_handles(runtime, event):
    runtime.fail_at = event
    cache = {}
    matrix = runtime.assembly()
    with pytest.raises(LinearSolveCapacityError) as raised:
        raw_amgx.solve_reduced_system_amgx_device(
            matrix, retry_attempts=({"label": "unused"},),
            cache_fixed_operator=True, retry_solver_cache=cache,
            scale_system=False, materialize_host_solution=False,
        )
    assert raised.value.amgx_attempt_count == 1
    assert all(solver.closed for solver in cache.values())
    assert_released(runtime)


def test_successful_host_adapter_still_returns_native_diagnostics_and_releases_objects(runtime):
    matrix = runtime.assembly()
    solution, info = cupy_backend.solve_pyamgx_csr(
        matrix, matrix.rhs, initial_guess=matrix.rhs, return_info=True
    )
    np.testing.assert_array_equal(solution, matrix.rhs)
    assert info["amgx_status"] == "SUCCESS"
    assert info["amgx_iterations"] == 0
    assert_released(runtime)


class SymbolicCode(Enum):
    NO_MEMORY = "no memory"


@pytest.mark.parametrize("error", (
    AMGXAllocationError("native allocation failed"),
    RuntimeError("cudaErrorMemoryAllocation"),
    RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"),
    type("OutOfMemoryError", (RuntimeError,), {})("allocation failed"),
    type("CUDARuntimeError", (RuntimeError,), {"status": 2})("allocation failed"),
    type("SymbolicAMGXError", (RuntimeError,), {"error_code": SymbolicCode.NO_MEMORY})("failed"),
))
def test_capacity_classification_preserves_wrapped_oom(error):
    outer = RuntimeError("backend failed")
    outer.__cause__ = error
    error.__context__ = outer  # Chained exceptions can contain cycles.
    assert is_amgx_capacity_error(outer)


@pytest.mark.parametrize("error", (
    RuntimeError("solver failed to converge"),
    RuntimeError("CUDA illegal memory access"),
    type("AMGXError", (RuntimeError,), {"error_code": 5})("CUDA failure"),
))
def test_other_backend_failures_are_not_classified_as_capacity(error):
    error.__context__ = error
    assert not is_amgx_capacity_error(error)


@pytest.mark.parametrize("stats", ("missing", "failing", "partial"))
def test_unavailable_memory_diagnostics_do_not_mask_capacity_failure(runtime, stats):
    def unavailable():
        raise RuntimeError("memory counters unavailable")

    if stats == "missing":
        del runtime.amgx.get_device_memory_stats
    elif stats == "failing":
        runtime.amgx.get_device_memory_stats = unavailable
    else:
        runtime.amgx.get_device_memory_stats = lambda: {"peak_live_bytes": 123}
    runtime.cp.cuda.runtime.memGetInfo = unavailable
    error = as_amgx_capacity_error(
        AMGXAllocationError("failed"), phase="solver setup", cp=runtime.cp, pyamgx=runtime.amgx
    )
    assert isinstance(error, LinearSolveCapacityError)
    assert "AMGX memory unavailable" in str(error)
    assert "memory counters unavailable" in error.memory["device_error"]
    if stats == "partial":
        assert error.memory["amgx"]["peak_live_bytes"] == 123
    assert as_amgx_capacity_error(error, phase="outer", cp=None, pyamgx=None) is error
