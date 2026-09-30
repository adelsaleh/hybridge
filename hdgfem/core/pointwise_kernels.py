"""Sampling kernels for Numba ``cfunc`` pointwise coefficients.

A pointwise coefficient is a compiled scalar function of the physical point,
the time and optionally a vector ``v`` of point data (DG field values,
gradients and parameters):

* ``float64(x, y, t)``: :data:`XYT_SIGNATURE`;
* ``float64(x, y, t, float64* v)``: :data:`VALUES_SIGNATURE`.

The kernels take the function as a first-class function argument, so they are
compiled once per signature and cached on disk; any number of user functions
of the same signature reuse that compilation. User functions are compiled by
:func:`hdgfem.core.pointwise.pointwise_coefficient`.
"""

from __future__ import annotations

import numpy as np
from numba import prange, types
from numba.extending import intrinsic

from hdgfem.runtime.optional import njit

XYT_SIGNATURE = types.float64(types.float64, types.float64, types.float64)
VALUES_SIGNATURE = types.float64(types.float64, types.float64, types.float64, types.CPointer(types.float64))


@intrinsic
def _data_pointer(typingctx, array):
    """``float64*`` to the first element of a C-contiguous float64 array."""
    signature = types.CPointer(types.float64)(array)

    def codegen(context, builder, sig, args):
        return context.make_array(sig.args[0])(context, builder, args[0]).data

    return signature, codegen


@njit(cache=True, parallel=True)
def sample_pointwise_xyt_kernel(function, x, y, t, out):
    """``out[k, q] = function(x[k, q], y[k, q], t)``."""
    for element in prange(x.shape[0]):
        for q in range(x.shape[1]):
            out[element, q] = function(x[element, q], y[element, q], t)


@njit(cache=True, parallel=True)
def sample_pointwise_values_kernel(function, x, y, t, values, params, out):
    """``out[k, q] = function(x, y, t, v)`` with ``v = [values[:, k, q], params]``."""
    count = values.shape[0]
    for element in prange(x.shape[0]):
        buffer = np.empty(count + params.shape[0], dtype=np.float64)
        for i in range(params.shape[0]):
            buffer[count + i] = params[i]
        pointer = _data_pointer(buffer)
        for q in range(x.shape[1]):
            for i in range(count):
                buffer[i] = values[i, element, q]
            out[element, q] = function(x[element, q], y[element, q], t, pointer)


__all__ = ["VALUES_SIGNATURE", "XYT_SIGNATURE", "sample_pointwise_values_kernel", "sample_pointwise_xyt_kernel"]
