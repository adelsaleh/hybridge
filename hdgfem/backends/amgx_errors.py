"""Shared AMGX capacity diagnostics and best-effort native resource cleanup."""

from __future__ import annotations

from typing import Any

from hdgfem.linalg.system import LinearSolveCapacityError


def is_amgx_capacity_error(exc: BaseException) -> bool:
    """Recognize AMGX/CUDA allocation failures through chained exceptions."""
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, LinearSolveCapacityError):
            return True
        code = getattr(current, "error_code", None)
        if getattr(code, "name", None) == "NO_MEMORY":
            return True
        try:
            if code is not None and int(code) == 7:  # AMGX_RC_NO_MEMORY
                return True
        except (TypeError, ValueError):
            pass
        kind = type(current).__name__.replace("_", "").lower()
        message = str(current).lower()
        compact_message = message.replace("_", "").replace(" ", "")
        if (
            "outofmemory" in kind
            or "outofmemory" in compact_message
            or "memoryallocation" in compact_message
            or "not enough memory" in message
            or (
                kind in {"cudaruntimeerror", "cudadrivererror"}
                and getattr(current, "status", None) == 2
            )
        ):
            return True
        pending.extend(
            nested
            for nested in (current.__cause__, current.__context__)
            if nested is not None
        )
    return False


def as_amgx_capacity_error(exc, *, phase: str, cp, pyamgx):
    """Capture available allocation counters without masking a capacity error."""
    if isinstance(exc, LinearSolveCapacityError):
        return exc
    if not is_amgx_capacity_error(exc):
        return None
    memory: dict[str, Any] = {}
    get_stats = getattr(pyamgx, "get_device_memory_stats", None)
    if get_stats is not None:
        try:
            memory["amgx"] = {key: int(value) for key, value in get_stats().items()}
        except Exception as stats_exc:
            memory["amgx_error"] = f"{type(stats_exc).__name__}: {stats_exc}"
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        memory["device"] = {
            "free_bytes": int(free_bytes),
            "used_bytes": int(total_bytes) - int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    except Exception as stats_exc:
        memory["device_error"] = f"{type(stats_exc).__name__}: {stats_exc}"
    gib = 1024.0 ** 3
    amgx = memory.get("amgx", {})
    device = memory.get("device")
    amgx_text = (
        f"live/reserved={amgx['live_bytes']/gib:.3f}/{amgx['reserved_bytes']/gib:.3f} GiB"
        if {"live_bytes", "reserved_bytes"} <= amgx.keys()
        else "unavailable"
    )
    device_text = (
        "unavailable"
        if device is None
        else "used/free/total="
        f"{device['used_bytes']/gib:.3f}/{device['free_bytes']/gib:.3f}/"
        f"{device['total_bytes']/gib:.3f} GiB"
    )
    return LinearSolveCapacityError(
        f"pyamgx-device capacity failure during {phase}: {exc}; "
        f"AMGX memory {amgx_text}; device memory {device_text}",
        backend="pyamgx-device",
        phase=phase,
        memory=memory,
    )


def destroy_amgx_objects(objects, *, suppress_errors: bool = False) -> None:
    """Attempt every native destroy, optionally propagating the first failure."""
    first_error = None
    for obj in objects:
        if obj is not None:
            try:
                obj.destroy()
            except AttributeError:
                pass
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error
