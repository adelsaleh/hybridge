"""Guard against native compilation when a script must only reuse existing JIT caches."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager


@contextmanager
def kernel_cache_only(enabled):
    """Allow loading existing JIT caches, but reject new native compilation."""
    with ExitStack() as stack:
        if enabled:
            from unittest.mock import patch

            def forbidden(*args, **kwargs):
                raise RuntimeError("Kernel cache miss: new compilation is disabled (--cached-kernels-only)")

            for name in ("numba.core.dispatcher._FunctionCompiler._compile_core",
                         "cupy.cuda.compiler.compile_using_nvrtc", "cupy.cuda.compiler.compile_using_nvcc"):
                stack.enter_context(patch(name, forbidden))
        yield
