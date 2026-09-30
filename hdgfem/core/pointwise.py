"""Compiled pointwise coefficients for the host Numba path.

:func:`pointwise_coefficient` compiles a scalar function of the physical point
with Numba (``cfunc``, cached on disk) and returns a
:class:`PointwiseCoefficient`, an :class:`~hdgfem.core.element_coefficients.ElementCoefficient`
that every ADR consumer already accepts (sampling, face samples,
post-processing). Evaluation runs in the parallel kernels of
:mod:`hdgfem.kernels.pointwise`, which are compiled once for all functions of
the same signature.

The function takes ``(x, y, t)`` or ``(x, y, t, v)``, where ``v`` holds, in
order, the values of ``fields``, the ``(d/dx, d/dy)`` pair of each field in
``gradients``, then ``params``::

    velocity = pointwise_coefficient(
        (lambda x, y, t, v: v[1]/max(v[0], v[2])*bx(x, y),
         lambda x, y, t, v: v[1]/max(v[0], v[2])*by(x, y)),
        space, fields=(n_star, gamma_star), params=(1e-8,), name="u* b_p")

:func:`pointwise_law` compiles the same kind of function into a
:class:`PointwiseLaw`, a plain ``(x, y)`` callable accepted wherever a
coefficient law is (for example the components of a diffusion tensor);
``v`` then holds only ``params``. :meth:`PointwiseCoefficient.bind` swaps the
fields, parameters or time without recompiling, so a time stepper compiles
once and rebinds every step. Functions that call other compiled functions
must reach them as module globals (``@njit`` functions defined at module
level): closures over compiled functions miss Numba's cache in every process.

Only NumPy/``math`` code compiles. SciPy functions, Python objects and other
libraries are rejected with an error asking to project the coefficient
first. Values that change between solves must be passed through ``params``,
``fields`` or the time argument: globals are frozen into the compiled code and
Numba's cache does not notice later changes, so functions reading numeric or
array globals are rejected. Closure variables are safe (the cache keys on
their values). Evaluation is host-only; the raw-CUDA path needs projected
coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Any
import math
import numbers
import warnings

import numpy as np

from hdgfem.precision import REAL_DTYPE

from .element_coefficients import ElementCoefficient
from .field_ops import field_gradient_at_ref, field_values_at_ref

_PROJECT_FIRST = ("Project it onto a DG space first (space.project_callable(...)) and pass the DGField, "
                  "or pass the plain Python callable, which is sampled with NumPy on the host.")


def _global_constants(function) -> dict:
    """Numeric or array globals read by ``function`` (frozen by Numba at compile time)."""
    code, scope = getattr(function, "__code__", None), getattr(function, "__globals__", {})
    names = set()
    stack = [code] if code is not None else []
    while stack:
        current = stack.pop()
        names.update(current.co_names)
        stack.extend(const for const in current.co_consts if inspect.iscode(const))
    def mathematical_constant(name, value):
        return any(getattr(module, name, None) is value or (isinstance(getattr(module, name, None), float)
                                                           and getattr(module, name) == value)
                   for module in (math, np))
    return {name: scope[name] for name in sorted(names) if name in scope
            and isinstance(scope[name], (numbers.Number, np.ndarray, np.generic))
            and not mathematical_constant(name, scope[name])}


def _compile(function, arity: int, name: str):
    """Compile one scalar component with the matching kernel signature."""
    from numba import cfunc

    from ..kernels.pointwise import VALUES_SIGNATURE, XYT_SIGNATURE
    frozen = _global_constants(function)
    if frozen:
        listed = ", ".join(f"{key}={value!r}" for key, value in frozen.items())
        raise ValueError(f"pointwise coefficient {name!r} reads the global value(s) {listed}; Numba freezes "
                         "globals into the compiled code and its cache does not notice later changes. Pass them "
                         "through params=(...) (read as v[...]) or capture them in a closure.")
    signature = XYT_SIGNATURE if arity == 3 else VALUES_SIGNATURE
    try:
        try:
            return cfunc(signature, cache=True)(function)
        except RuntimeError as error:
            if "cannot cache" not in str(error):
                raise
            # Defined outside a file (REPL, notebook cell, exec): no disk cache is possible.
            warnings.warn(f"pointwise coefficient {name!r} is not defined in a file, so Numba cannot cache it; "
                          "it is compiled again in every process", stacklevel=4)
            return cfunc(signature)(function)
    except Exception as error:  # noqa: BLE001 - every compilation failure gets the same guidance
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        detail = next((line for line in lines if not line.startswith("Failed in")), lines[0] if lines else "")
        detail = detail if len(detail) <= 160 else detail[:157] + "..."
        raise TypeError(f"pointwise coefficient {name!r} cannot be compiled by Numba "
                        f"({type(error).__name__}: {detail}). Numba compiles NumPy and "
                        f"math code only; SciPy functions, Python objects and other libraries are not supported. "
                        + _PROJECT_FIRST) from error


def _parameters(params) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_1d(np.asarray(params, dtype=REAL_DTYPE)).ravel())


def _check_fields(mesh, fields, name: str) -> None:
    for field in fields:
        if field.space.mesh.triangulation is not mesh.triangulation:
            raise ValueError(f"pointwise coefficient {name!r}: every field must live on the coefficient's mesh")


def _arity(function, name: str) -> int:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        raise TypeError(f"pointwise coefficient {name!r} must be a Python function") from None
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) not in (3, 4) or len(positional) != len(list(parameters)):
        raise TypeError(f"pointwise coefficient {name!r} must take (x, y, t) or (x, y, t, v)")
    return len(positional)


@dataclass(frozen=True, eq=False)
class PointwiseCoefficient(ElementCoefficient):
    """Element coefficient evaluated by Numba-compiled pointwise functions (one per component).

    Build it with :func:`pointwise_coefficient`. ``time`` is used when a
    consumer samples without a time; :meth:`at_time` returns a copy bound to
    another time without recompiling.
    """

    functions: tuple = ()
    fields: tuple = ()
    gradients: tuple = ()
    params: np.ndarray | None = None
    time: float = 0.
    arity: int = 3

    def __post_init__(self):
        """Validate and bind the evaluator."""
        super().__post_init__()
        if len(self.functions) != self.components:
            raise ValueError("pointwise coefficient needs one compiled function per component")
        object.__setattr__(self, "function", self._evaluate)

    def at_time(self, t: float) -> "PointwiseCoefficient":
        """The same compiled coefficient bound to time ``t``."""
        return replace(self, time=float(t))

    def bind(self, *, fields=None, gradients=None, params=None, time=None) -> "PointwiseCoefficient":
        """The same compiled functions with new point data or time (no recompilation).

        The layout of ``v`` must not change: the same number of fields,
        gradient fields and parameters.
        """
        changes = {key: tuple(value) for key, value in (("fields", fields), ("gradients", gradients))
                   if value is not None}
        if params is not None:
            changes["params"] = _parameters(params)
        if time is not None:
            changes["time"] = float(time)
        bound = replace(self, **changes)
        if (len(bound.fields), len(bound.gradients), bound.params.size) != (
                len(self.fields), len(self.gradients), self.params.size):
            raise ValueError(f"pointwise coefficient {self.name!r}: bind must keep the number of fields, "
                             "gradients and parameters")
        _check_fields(self.mesh, bound.fields + bound.gradients, self.name)
        return bound

    def _point_values(self, points) -> np.ndarray:
        """``(count, K, n)`` field values, then gradient pairs, at reference points."""
        rows = [field_values_at_ref(field, points) for field in self.fields]
        for field in self.gradients:
            rows.extend(field_gradient_at_ref(field, points))
        if not rows:
            return np.empty((0, self.mesh.num_tri, points.shape[0]), dtype=REAL_DTYPE)
        return np.ascontiguousarray(np.stack(rows), dtype=REAL_DTYPE)

    def _evaluate(self, points, *, xp=np, t=None):
        """Values at ``(n, 2)`` reference points on every element, ``(K, n[, c])``."""
        if xp is not np:
            raise TypeError(f"pointwise coefficient {self.name!r} is evaluated by host Numba kernels; the device "
                            "path needs a projected coefficient (space.project_callable(...)).")
        from ..kernels.pointwise import sample_pointwise_values_kernel, sample_pointwise_xyt_kernel
        points = np.ascontiguousarray(np.asarray(points, dtype=REAL_DTYPE).reshape(-1, 2))
        mapped = self.mesh.map_reference_points(points)
        x, y = np.ascontiguousarray(mapped[..., 0]), np.ascontiguousarray(mapped[..., 1])
        time = float(self.time if t is None else t)
        values = self._point_values(points) if self.arity == 4 else None
        outputs = []
        for function in self.functions:
            out = np.empty(x.shape, dtype=REAL_DTYPE)
            if self.arity == 3:
                sample_pointwise_xyt_kernel(function, x, y, time, out)
            else:
                sample_pointwise_values_kernel(function, x, y, time, values, self.params, out)
            outputs.append(out)
        return outputs[0] if self.components == 1 else np.stack(outputs, axis=-1)


def pointwise_coefficient(function, mesh, *, fields=(), gradients=(), params=(), time: float = 0.,
                          name: str = "coefficient") -> PointwiseCoefficient:
    """Compile ``function`` (or a tuple of component functions) into a :class:`PointwiseCoefficient`.

    ``mesh`` is a mesh or a space on it; ``fields`` and ``gradients`` are DG
    fields on the same mesh. Functions take ``(x, y, t)`` or, when point data
    is needed, ``(x, y, t, v)`` with ``v = [fields..., (dx, dy) per gradient
    field..., params...]``. Compilation errors (for example SciPy or Python
    objects in the function) raise :class:`TypeError` asking to project the
    coefficient first.
    """
    from ..kernels import NUMBA_AVAILABLE
    if not NUMBA_AVAILABLE:
        raise RuntimeError("pointwise coefficients need Numba. " + _PROJECT_FIRST)
    mesh = getattr(mesh, "mesh", mesh)
    functions = tuple(function) if isinstance(function, (tuple, list)) else (function,)
    fields, gradients, params = tuple(fields), tuple(gradients), _parameters(params)
    _check_fields(mesh, fields + gradients, name)
    arities = {_arity(component, name) for component in functions}
    if len(arities) != 1:
        raise TypeError(f"pointwise coefficient {name!r}: all components must take the same arguments")
    arity = arities.pop()
    if arity == 3 and (fields or gradients or params.size):
        raise TypeError(f"pointwise coefficient {name!r} has fields/gradients/params, so it must take (x, y, t, v)")
    compiled = tuple(_compile(component, arity, f"{name}[{index}]" if len(functions) > 1 else name)
                     for index, component in enumerate(functions))
    return PointwiseCoefficient(None, mesh, len(functions), name, functions=compiled, fields=fields,
                                gradients=gradients, params=params, time=float(time), arity=arity)


@dataclass(frozen=True, eq=False)
class PointwiseLaw:
    """Compiled scalar law of the physical point, called as ``law(x, y)`` on host arrays.

    Build it with :func:`pointwise_law`. Evaluation runs in the parallel
    kernels of :mod:`hdgfem.kernels.pointwise` at ``time``; the result has
    the broadcast shape of ``x`` and ``y``.
    """

    function: Any
    params: np.ndarray
    time: float = 0.
    arity: int = 3
    name: str = "law"

    def __call__(self, x, y):
        """Values at host points ``(x, y)`` (broadcast)."""
        if hasattr(x, "__cuda_array_interface__") or hasattr(y, "__cuda_array_interface__"):
            raise TypeError(f"pointwise law {self.name!r} is evaluated by host Numba kernels; the device "
                            "path needs a projected coefficient (space.project_callable(...)).")
        from ..kernels.pointwise import sample_pointwise_values_kernel, sample_pointwise_xyt_kernel
        x, y = np.broadcast_arrays(np.asarray(x, dtype=REAL_DTYPE), np.asarray(y, dtype=REAL_DTYPE))
        shape = x.shape
        if x.size == 0:
            return np.empty(shape, dtype=REAL_DTYPE)
        rows = shape[0] if x.ndim > 1 else 1
        x, y = np.ascontiguousarray(x).reshape(rows, -1), np.ascontiguousarray(y).reshape(rows, -1)
        out = np.empty(x.shape, dtype=REAL_DTYPE)
        if self.arity == 3:
            sample_pointwise_xyt_kernel(self.function, x, y, self.time, out)
        else:
            values = np.empty((0,) + x.shape, dtype=REAL_DTYPE)
            sample_pointwise_values_kernel(self.function, x, y, self.time, values, self.params, out)
        return out.reshape(shape)

    def at_time(self, t: float) -> "PointwiseLaw":
        """The same compiled law bound to time ``t``."""
        return replace(self, time=float(t))


def pointwise_law(function, *, params=(), time: float = 0., name: str = "law") -> PointwiseLaw:
    """Compile ``function`` into a :class:`PointwiseLaw` usable as an ``(x, y)`` coefficient callable.

    ``function`` takes ``(x, y, t)`` or, with ``params``, ``(x, y, t, v)``
    where ``v`` holds the parameters. Compilation errors raise
    :class:`TypeError` as for :func:`pointwise_coefficient`.
    """
    from ..kernels import NUMBA_AVAILABLE
    if not NUMBA_AVAILABLE:
        raise RuntimeError("pointwise laws need Numba. " + _PROJECT_FIRST)
    params = _parameters(params)
    arity = _arity(function, name)
    if arity == 3 and params.size:
        raise TypeError(f"pointwise law {name!r} has params, so it must take (x, y, t, v)")
    return PointwiseLaw(_compile(function, arity, name), params, float(time), arity, name)


__all__ = ["PointwiseCoefficient", "PointwiseLaw", "pointwise_coefficient", "pointwise_law"]
