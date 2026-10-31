"""Array-only checks for application containers; no numerical solve or stepping."""
from dataclasses import dataclass
import numpy as np
import pytest
from hybridge.runtime.precision import REAL_DTYPE, PIPELINE_AUDIT, audit_arrays


def test_precision_audit_walks_application_dataclasses_with_slots(monkeypatch):
    monkeypatch.setenv('HYBRIDGE_PRECISION_AUDIT', '1')

    @dataclass(slots=True)
    class State:
        values: object

    values = np.ones(3, dtype=REAL_DTYPE)
    audit_arrays('application-container', State(values))
    assert PIPELINE_AUDIT['application-container']['host_arrays'] == 1
    wrong = np.float32 if np.dtype(REAL_DTYPE) == np.dtype(np.float64) else np.float64
    with pytest.raises(TypeError, match='values has'):
        audit_arrays('application-container', State(values.astype(wrong)))


def test_precision_audit_does_not_walk_dataclass_types(monkeypatch):
    monkeypatch.setenv('HYBRIDGE_PRECISION_AUDIT', '1')

    @dataclass
    class Definition:
        values: object

    audit_arrays('application-definition', Definition)
    assert PIPELINE_AUDIT['application-definition']['host_arrays'] == 0
