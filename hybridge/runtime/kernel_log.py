"""Say in the log when a CUDA kernel first runs, for a runner's high verbosity.

A kernel's first launch also compiles it or loads it from CuPy's disk cache, and the first use of a
tuned workload on a machine also measures its candidates. Those costs land in whichever timed row is
running, so these lines say why that row is slow and whether the cost comes back on the next run.

:func:`log_kernel_first_runs` replaces ``cupy.RawKernel`` and ``cupy.RawModule`` by subclasses that report
their first launch, which covers every compile site at once. Nothing is replaced until a caller asks,
and kernels made before that are not reported.
"""
from __future__ import annotations

import time

from hybridge.runtime import optional
from hybridge.runtime.logging import log_line

_enabled = False
_installed = False
_reported: set = set()  # solvers can rebuild a kernel object from the same code, which is not a first run


def log_kernel_first_runs(enabled: bool = True) -> None:
    """Report kernels and tuned workloads on their first use from now on, or stop reporting."""
    global _enabled
    _enabled = bool(enabled)
    if _enabled:
        _install()


def _report_first_run(name, code, seconds: float) -> None:
    """Log the first run of kernel ``name`` and its ``seconds``; nothing is logged while reporting is off, and a kernel with the same name and code is logged once per process."""
    key = (name, hash(code))
    if _enabled and key not in _reported:
        _reported.add(key)
        log_line(f"[kernel] first run: {name} ({seconds:.4f} s, compiling or loading included)")


def _workload_label(identity) -> str:
    """Name a tuned workload from the identity that its selection already keeps."""
    head = identity.get("operation") or identity.get("family") or "kernel"
    details = [f"{key}={identity[key]}" for key in ("phase", "stage", "order", "format", "policy")
               if identity.get(key) is not None]
    return f"{head} ({', '.join(details)})" if details else str(head)


def report_selection(identity, choice: str, source: str, seconds: float, *, candidates: int = 0) -> None:
    """Report how a tuned workload got its kernel; ``source`` is that of ``select_variant``.

    A selection made earlier in this process (``process``) is not a first use and says nothing.
    """
    if not _enabled or source == "process":
        return
    label = _workload_label(identity)
    if source == "measured":
        log_line(f"[kernel] first time on this machine: {label}: measured {candidates} candidates "
                 f"in {seconds:.4f} s, chose {choice}; later runs reuse this")
    elif source == "disk":
        log_line(f"[kernel] tuned on an earlier run: {label}: {choice}, ready in {seconds:.4f} s")
    else:
        log_line(f"[kernel] not measured ({source}): {label}: {choice}, ready in {seconds:.4f} s")


class _FirstRunFunction:
    """Forward one function of a CUDA module, reporting its first launch."""

    def __init__(self, function, name: str, code, compile_seconds: float):
        """Wrap ``function`` of a module; ``compile_seconds``, the module compile time, is added to the time of its first launch."""
        self._first_run_function = function
        self._first_run_name = name
        self._first_run_code = code
        self._first_run_seconds = compile_seconds
        self._first_run_pending = True

    def __getattr__(self, name):
        """Return the attribute ``name`` of the wrapped function."""
        return getattr(self._first_run_function, name)

    def __setattr__(self, name, value):
        """Set the ``_first_run_*`` attributes on this wrapper and every other attribute on the wrapped function."""
        if name.startswith("_first_run_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._first_run_function, name, value)

    def __call__(self, *args, **kwargs):
        """Launch the wrapped function; on the first launch, also log its time plus the compile time."""
        if not self._first_run_pending:
            return self._first_run_function(*args, **kwargs)
        self._first_run_pending = False
        start = time.perf_counter()
        result = self._first_run_function(*args, **kwargs)
        _report_first_run(self._first_run_name, self._first_run_code,
                          self._first_run_seconds + time.perf_counter() - start)
        return result


def _install() -> None:
    """Replace CuPy's two kernel classes once, if CuPy is present."""
    global _installed
    cupy = optional.cp
    if _installed or cupy is None:
        return

    class FirstRunKernel(cupy.RawKernel):
        """A raw kernel that reports its first launch, with any earlier explicit compile."""

        _first_run_pending = True
        _first_run_seconds = 0.0

        def compile(self, *args, **kwargs):
            """Compile the kernel and add the compile time to the time logged at its first launch."""
            start = time.perf_counter()
            result = super().compile(*args, **kwargs)
            self._first_run_seconds += time.perf_counter() - start
            return result

        def __call__(self, *args, **kwargs):
            """Launch the kernel; on the first launch, also log its time plus any earlier compile time."""
            if not self._first_run_pending:
                return super().__call__(*args, **kwargs)
            self._first_run_pending = False
            start = time.perf_counter()
            result = super().__call__(*args, **kwargs)
            _report_first_run(self.name, self.code, self._first_run_seconds + time.perf_counter() - start)
            return result

    class FirstRunModule(cupy.RawModule):
        """A raw module whose functions report their first launch; its compile joins the first function's."""

        _first_run_compile_seconds = 0.0

        def compile(self, *args, **kwargs):
            """Compile the module and add the compile time to the time logged with the next function taken from it."""
            start = time.perf_counter()
            result = super().compile(*args, **kwargs)
            self._first_run_compile_seconds += time.perf_counter() - start
            return result

        def get_function(self, name):
            """Return the function ``name`` of the module, wrapped to log its first launch together with the module compile time so far; that compile time then restarts from zero."""
            start = time.perf_counter()
            function = super().get_function(name)
            seconds = self._first_run_compile_seconds + time.perf_counter() - start
            self._first_run_compile_seconds = 0.0
            return _FirstRunFunction(function, name, self.code, seconds)

    cupy.RawKernel, cupy.RawModule = FirstRunKernel, FirstRunModule
    _installed = True
