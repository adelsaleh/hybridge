"""Host policy checks; numerical pipeline checks live in test_advection_tensor_auto."""
import pytest
import numpy as np
from types import SimpleNamespace

from hybridge.transport.pipeline_tuning import candidate_pipelines
from hybridge.solvers.advection_reaction import _resolve_local_assembly


@pytest.mark.parametrize("rich", [True, False])
def test_all_architectures_consider_equivalent_pipeline_families(rich):
    candidates = candidate_pipelines(order=6, requested="auto", tensor=True, rich_fp64=rich)
    assert candidates[0].assembly == "fused"
    assert {c.assembly for c in candidates} == {"fused", "split3", "tensor"}
    assert {c.local_solver for c in candidates if c.assembly == "tensor"} == {"cublas", "coop", "fused", "magma"}
    assert len(candidates) <= 8


def test_explicit_launch_and_tensor_engine_choices_constrain_candidates():
    candidates = candidate_pipelines(order=6, requested=32, tensor=True, rich_fp64=True,
                                    contraction_override="cublas", solver_override="coop")
    assert all(c.block == 32 for c in candidates)
    assert all((c.contraction, c.local_solver) == ("cublas", "coop")
               for c in candidates if c.assembly == "tensor")
    assert all(c.assembly != "tensor" for c in candidate_pipelines(
        order=6, requested="auto", tensor=False, rich_fp64=True))
    without_cutensor = candidate_pipelines(order=6, requested="auto", tensor=True, rich_fp64=True,
                                          cutensor_available=False)
    assert {c.local_solver for c in without_cutensor if c.assembly == "tensor"} == {"cublas", "coop", "fused", "magma"}
    assert all(c.contraction == "cublas" for c in without_cutensor)


@pytest.mark.parametrize("settings,expected", [
    ({}, "auto"), ({"bsr_output": False}, "fused"),
    ({"cache_operator": True}, "fused"), ({"raw_lu_mode": "safe"}, "fused"),
])
def test_auto_defers_only_compatible_workflows(settings, expected):
    options = dict(bsr_output=True, raw_lu_mode=None)
    options.update(settings)
    assert _resolve_local_assembly(None, "legendre-modal", "auto", **options) == expected
    assert _resolve_local_assembly(None, "legendre-modal", "split3", **options) == "split3"


def test_production_memory_limits_exclude_staged_pipelines_before_sampling(monkeypatch):
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.transport import pipeline_tuning, tsle_bsr, tsle_tensor, tuning_sample
    host = DGSpace(rectangle_mesh(2, 2), 6, basis_type="dub_orth")
    trace = host.trace_space("legendre-modal")
    cspace = SimpleNamespace(host=host, el_dof=host.el_dof, edg_dof=trace.edg_dof)
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (10**8, 2*10**8))),
                         get_default_memory_pool=lambda: SimpleNamespace(free_bytes=lambda: 0))
    monkeypatch.setattr(pipeline_tuning, "device_signature", lambda cp: {
        "sharedMemPerBlockOptin": 100000, "maxThreadsPerBlock": 1024, "fp32_fp64_ratio": 2})
    monkeypatch.setattr(tsle_bsr, "split3_memory_sizes", lambda *args: (8*10**7, 10**8))
    monkeypatch.setattr(tsle_tensor, "tensor_memory_sizes", lambda *args: (7*10**7, 10**8))
    monkeypatch.setattr(tsle_tensor, "require_cutensor", lambda: None)
    monkeypatch.setattr(tuning_sample, "sample_raw_options", lambda *a, **kw: pytest.fail("allocated sample"))
    def select(**kwargs):
        assert all(name.startswith("fused:") for name in kwargs["candidates"])
        return kwargs["baseline"], None, {}
    monkeypatch.setattr(pipeline_tuning, "select_variant", select)
    options = dict(cspace=cspace, trace_ref=trace, beta_coeffs=np.zeros((2, host.mesh.num_tri, host.el_dof)),
                   beta_face_basis=None, block_size="auto", matrix_format="bsr", reaction_is_scalar=True,
                   use_sparse_advection=True, mass_is_diagonal=True, advection_stabilization=None,
                   zero_boundary_flux=True, cache_local_response=True)
    selected, _ = pipeline_tuning.select_pipeline(cp, options)
    assert selected.assembly == "fused"
