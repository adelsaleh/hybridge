"""Compatibility imports for the shared runtime CUDA tuner.

The implementation lives below linalg so assembly and solver kernels use the
same device/workload cache without introducing a reverse layer dependency.
"""
from hybridge.runtime.cuda_tuning import (
    CACHE_VERSION, CandidateUnavailable, KernelConfiguration, TuningBudget,
    cache_directory, clear_tuning_cache, device_signature, event_seconds,
    fingerprint, legal_configuration, select_configuration, select_variant,
    tuning_budget, tuning_mode,
)
