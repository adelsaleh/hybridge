"""First-run lines of CUDA kernels: once per kernel, nothing unless asked, kernel behavior unchanged."""
import types

import pytest

from hybridge.runtime import kernel_log, optional


class _Kernel:
    def __init__(self, code, name, **options):
        self.code, self.name, self.calls, self.compiled = code, name, [], 0
        self.max_dynamic_shared_size_bytes = 0

    def compile(self):
        self.compiled += 1

    def __call__(self, grid, block, args, **options):
        self.calls.append((grid, block, args, options))
        return "launched"


class _Function:
    ptr = 1234

    def __init__(self):
        self.calls = []
        self.max_dynamic_shared_size_bytes = 0

    def __call__(self, grid, block, args):
        self.calls.append((grid, block, args))
        return "launched"


class _Module:
    def __init__(self, *, code, **options):
        self.code = code

    def compile(self):
        pass

    def get_function(self, name):
        return _Function()


@pytest.fixture
def fake_cupy(monkeypatch):
    cupy = types.SimpleNamespace(RawKernel=_Kernel, RawModule=_Module)
    monkeypatch.setattr(optional, "cp", cupy)
    monkeypatch.setattr(kernel_log, "_installed", False)
    monkeypatch.setattr(kernel_log, "_enabled", False)
    monkeypatch.setattr(kernel_log, "_reported", set())
    return cupy


def test_nothing_changes_until_the_log_is_asked_for(fake_cupy):
    kernel_log.log_kernel_first_runs(False)
    assert fake_cupy.RawKernel is _Kernel and fake_cupy.RawModule is _Module


def test_a_kernel_reports_only_its_first_launch_and_forwards_everything(fake_cupy, capsys):
    kernel_log.log_kernel_first_runs()
    kernel = fake_cupy.RawKernel("code", "my_kernel", options=("--std=c++11",))
    kernel.compile()
    assert kernel.compiled == 1 and capsys.readouterr().out == ""
    assert kernel((1,), (64,), ("a",), shared_mem=8) == "launched"
    assert "[kernel] first run: my_kernel (" in capsys.readouterr().out
    assert kernel((2,), (64,), ("b",)) == "launched"
    assert capsys.readouterr().out == ""
    assert kernel.calls == [((1,), (64,), ("a",), {"shared_mem": 8}), ((2,), (64,), ("b",), {})]


def test_a_module_function_reports_its_first_launch_and_forwards_attributes(fake_cupy, capsys):
    kernel_log.log_kernel_first_runs()
    function = fake_cupy.RawModule(code="code").get_function("my_function")
    assert function.ptr == 1234
    function.max_dynamic_shared_size_bytes = 77
    assert function._first_run_function.max_dynamic_shared_size_bytes == 77
    assert capsys.readouterr().out == ""
    assert function((1,), (64,), ("a",)) == "launched"
    assert "[kernel] first run: my_function (" in capsys.readouterr().out
    function((1,), (64,), ("b",))
    assert capsys.readouterr().out == ""


def test_a_kernel_object_rebuilt_from_the_same_code_is_not_a_first_run(fake_cupy, capsys):
    kernel_log.log_kernel_first_runs()
    fake_cupy.RawKernel("code", "my_kernel")((1,), (64,), ())
    assert "first run: my_kernel" in capsys.readouterr().out
    fake_cupy.RawKernel("code", "my_kernel")((1,), (64,), ())
    assert capsys.readouterr().out == ""
    fake_cupy.RawKernel("other code", "my_kernel")((1,), (64,), ())
    assert "first run: my_kernel" in capsys.readouterr().out


def test_stopping_the_log_silences_kernels_made_while_it_was_on(fake_cupy, capsys):
    kernel_log.log_kernel_first_runs()
    kernel = fake_cupy.RawKernel("code", "my_kernel")
    kernel_log.log_kernel_first_runs(False)
    kernel((1,), (64,), ())
    assert capsys.readouterr().out == ""


def test_no_cupy_is_not_an_error(monkeypatch):
    monkeypatch.setattr(optional, "cp", None)
    monkeypatch.setattr(kernel_log, "_installed", False)
    monkeypatch.setattr(kernel_log, "_enabled", False)
    kernel_log.log_kernel_first_runs()


def test_real_kernels_still_compile_and_run(monkeypatch, capsys):
    cp = optional.require_cupy_device()
    monkeypatch.setattr(cp, "RawKernel", cp.RawKernel)  # restored after the test
    monkeypatch.setattr(cp, "RawModule", cp.RawModule)
    monkeypatch.setattr(kernel_log, "_installed", False)
    monkeypatch.setattr(kernel_log, "_enabled", False)
    monkeypatch.setattr(kernel_log, "_reported", set())
    from hybridge.runtime.cuda_tuning import kernel_attributes

    kernel_log.log_kernel_first_runs()
    source = 'extern "C" __global__ void fill(double* out) { out[blockIdx.x*blockDim.x+threadIdx.x] = 2.; }'
    output = cp.zeros(8)
    kernel = cp.RawKernel(source, "fill")
    kernel((1,), (8,), (output,))
    assert float(output.sum()) == 16. and "first run: fill" in capsys.readouterr().out
    # The module's copy has other code around it, so that it counts as a kernel of its own.
    function = cp.RawModule(code="// module\n" + source).get_function("fill")
    assert kernel_attributes(cp, function)["max_threads_per_block"] > 0
    output.fill(0)
    function((1,), (8,), (output,))
    assert float(output.sum()) == 16. and "first run: fill" in capsys.readouterr().out
