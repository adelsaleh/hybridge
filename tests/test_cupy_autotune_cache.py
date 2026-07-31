from __future__ import annotations

import json

from hdgfem.backends.cupy_autotune import (
    AUTOTUNE_KERNEL_ABI_VERSION,
    CUDADeviceFingerprint,
    FaceDenseAutotuneKey,
    FaceDenseAutotuneResult,
    KernelCandidateTiming,
    PersistentFaceDenseAutotuneCache,
)


def _key(*, order: int = 2, mode: str = "eliminate") -> FaceDenseAutotuneKey:
    return FaceDenseAutotuneKey(
        kernel_abi_version=AUTOTUNE_KERNEL_ABI_VERSION,
        device=CUDADeviceFingerprint(
            device_name="NVIDIA T600",
            compute_capability="7.5",
            total_global_memory=4 * 2**30,
            driver_version=13000,
            runtime_version=12080,
        ),
        dtype="float64",
        num_rows=12160,
        num_slots=5,
        block_size=order + 1,
        boundary_mode=mode,
        polynomial_order=order,
        local_solver="cublas_inverse",
        operator_implementations=("raw", "raw_fused"),
        asm_applications=("raw", "fused"),
    )


def _result() -> FaceDenseAutotuneResult:
    return FaceDenseAutotuneResult(
        device_name="NVIDIA T600",
        dtype="float64",
        num_dofs=36480,
        block_size=3,
        operator_choice="raw",
        asm_choice="fused",
        operator_candidates=(
            KernelCandidateTiming(
                "raw", 0.10, 0.09, 1024, 0.0,
                mean_ms=0.11,
                standard_deviation_ms=0.01,
                p90_ms=0.12,
                repeats=50,
            ),
        ),
        asm_candidates=(
            KernelCandidateTiming("fused", 0.07, 0.06, 512, 1.0e-16),
        ),
    )


def test_autotune_key_is_deterministic_and_context_sensitive() -> None:
    first = _key()
    second = FaceDenseAutotuneKey.from_dict(first.to_dict())
    assert first == second
    assert first.cache_id == second.cache_id
    assert first.cache_id != _key(order=3).cache_id
    assert first.cache_id != _key(mode="penalty").cache_id


def test_autotune_result_round_trip_preserves_extended_statistics() -> None:
    original = _result()
    restored = FaceDenseAutotuneResult.from_dict(original.to_dict())
    assert restored == original
    assert restored.operator_candidates[0].p90_ms == 0.12
    assert restored.operator_candidates[0].repeats == 50


def test_persistent_cache_round_trip_and_overwrite(tmp_path) -> None:
    path = tmp_path / "autotune.json"
    cache = PersistentFaceDenseAutotuneCache(path)
    key = _key()
    result = _result()

    assert cache.get(key) is None
    cache.put(key, result)
    assert cache.entry_count() == 1
    assert cache.get(key) == result

    replacement = FaceDenseAutotuneResult(
        **{**result.__dict__, "operator_choice": "raw_fused"}
    )
    cache.put(key, replacement)
    assert cache.entry_count() == 1
    assert cache.get(key) == replacement

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert key.cache_id in payload["entries"]


def test_cache_ignores_corrupt_or_incompatible_files(tmp_path) -> None:
    path = tmp_path / "autotune.json"
    path.write_text("not-json")
    cache = PersistentFaceDenseAutotuneCache(path)
    assert cache.get(_key()) is None

    path.write_text(json.dumps({"schema_version": 999, "entries": {}}))
    assert cache.get(_key()) is None


def test_cache_rejects_entry_for_different_requested_candidates(tmp_path) -> None:
    cache = PersistentFaceDenseAutotuneCache(tmp_path / "autotune.json")
    key = _key()
    cache.put(key, _result())
    changed = FaceDenseAutotuneKey(
        **{
            **key.__dict__,
            "operator_implementations": ("raw_fused",),
        }
    )
    assert cache.get(changed) is None
