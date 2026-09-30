"""Device coefficient sampling for raw-CUDA ADR assembly.

``prepare_adr_data_cupy`` is the device counterpart of
``assembly.advection_diffusion_reaction.prepare_adr_data(dense_local_matrices=False)``:
it returns the same :class:`ADRPreparedData` fields, but every per-element
sample table (source moments, reaction and beta on volume points, beta.n,
advection and diffusion tau, their sum and gamma on face points) is a CuPy
array computed from the cached device space. They feed
``adr_tensor_raw_cuda.assemble_tensor_operator`` without a host round trip; the
raw-CUDA solver times this step as part of assembly.

Sampling reuses the shared device samplers of ``backends.coefficients_cupy``
and the ``device=True`` paths of the shared policies (for example
``normalize_diffusion_stabilization``, whose global-length and inverse-h rules
need n^T kappa n on every face point of a variable tensor). Advection tau is
evaluated on the device for the upwind family (upwind, ``ScaledUpwind``,
Lax--Friedrichs) and explicit scalar/field/face forms. Conflict repair and
callables that raise ``TypeError`` on CuPy input use host sampling and upload, so
results match ``prepare_adr_data`` up to round-off.
"""
from __future__ import annotations

import time

import numpy as np

from hdgfem.core.element_coefficients import ElementCoefficient
from hdgfem.core.space import DGSpace, DGTraceSpace, VectorDGField
from hdgfem.core.device import as_cupy_space
from hdgfem.runtime.optional import require_cupy


def _elapsed(cp, start: float) -> float:
    """Synchronize the stream so a stage's wall time includes its device work."""
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter() - start


def _beta_samples(cp, beta, space: DGSpace, trace_ref: DGTraceSpace, t=None):
    """Return device beta.n on faces (K, 3, nfq) and beta on volume points (K, nq, 2).

    As on the host, callable beta enters beta.n exactly and the volume samples
    through its L2 projection into ``space``; DG beta uses the same field for both;
    an ``ElementCoefficient`` is evaluated directly at both point sets.
    """
    from hdgfem.hdg import condensation as hdg
    from hdgfem.assembly.advection_diffusion_reaction import element_beta_samples

    if isinstance(beta, ElementCoefficient):
        return element_beta_samples(beta, space, trace_ref, xp=cp, t=t)
    from hdgfem.hdg.coefficients import _is_callable_beta
    from hdgfem.hdg.coefficients_device import (
            field_on_faces_cupy,
            field_on_volume_cupy,
            mapped_face_points_cupy,
            volume_samples_cupy,
        )
    from hdgfem.core.device import _normalize_cupy_values

    cspace = as_cupy_space(space)
    if _is_callable_beta(beta):
        num_elements, nfq = space.mesh.num_tri, trace_ref.weights.size
        points = mapped_face_points_cupy(space, trace_ref)  # (K, 3*nfq, 2), flat q*3 + face
        face_values, volume_values = [], []
        for component, label in zip(beta, ("beta[0]", "beta[1]")):
            flat = _normalize_cupy_values(component(points[..., 0], points[..., 1]), num_elements, 3 * nfq, label)
            face_values.append(flat.reshape(num_elements, nfq, 3).transpose(0, 2, 1))
            coefficients = volume_samples_cupy(component, space, label=label) @ cspace.quad_data.projection_operator.T
            volume_values.append(coefficients @ cspace.quad_data.bas_of_quads)
    else:
        field = beta if isinstance(beta, VectorDGField) else hdg.as_vector_field(beta, space)
        if field.dim != 2:
            raise ValueError("advection field must have two components")
        face_values = [field_on_faces_cupy(component, space, trace_ref) for component in field.components]
        volume_values = [field_on_volume_cupy(component, space) for component in field.components]
    normals = cspace.mesh.normals
    normal = face_values[0] * normals[:, :, 0, None] + face_values[1] * normals[:, :, 1, None]
    return cp.ascontiguousarray(normal), cp.ascontiguousarray(cp.stack(volume_values, axis=-1))


def _source_moments(cp, source, space: DGSpace, t=None):
    """Element source moments, shape (K, NEL).

    Device arrays (moments or volume-quadrature values) and ``ElementCoefficient``
    sources stay on the device; other non-callable forms use the host sampler.
    """
    from hdgfem.hdg import condensation as hdg
    from hdgfem.core.space import DGField
    from hdgfem.backends.advection_cuda import source_moments_cupy, source_moments_from_values_cupy

    if isinstance(source, ElementCoefficient):
        values = source.volume_values(space, xp=cp, t=t)
        return cp.ascontiguousarray(source_moments_from_values_cupy(values, as_cupy_space(space)))
    same_space_field = isinstance(source, DGField) and (
        source.space is space or source.constant_value is not None)
    if same_space_field or isinstance(source, cp.ndarray) or (
            callable(source) and not isinstance(source, DGField)):
        return cp.ascontiguousarray(source_moments_cupy(source, as_cupy_space(space)))
    return cp.asarray(hdg.source_moments(source, space))


def _host_fallback(cp, device, host, timings: dict, key: str):
    """Run ``device()``; on ``TypeError`` (CuPy-incompatible callable) upload ``host()``."""
    try:
        return device()
    except TypeError:
        timings[f"{key}.host_fallback"] = 1.0
        result = host()
        if isinstance(result, tuple):
            return tuple(cp.asarray(value) for value in result)
        return cp.asarray(result)


def prepare_adr_data_cupy(
        source,
        reaction,
        beta,
        space: DGSpace,
        *,
        diffusion=1.0,
        advection_stabilization=None,
        diffusion_stabilization="global_length",
        diffusion_penalty_constant: float = 1.0,
        trace_space: DGTraceSpace | None = None,
        t=None,
        timings: dict[str, float] | None = None,
        tau_diffusion=None,
):
    """Sample ADR coefficients on the device; see the module docstring.

    Stage wall times (synchronized) are added to ``timings`` under
    ``raw.coefficients.*``; ``preparation_seconds`` holds their sum.
    ``diffusion_stabilization_law`` keeps a callable/DG law itself and otherwise
    the device ``tau_diffusion`` table, which postprocessing resamples.
    A precomputed ``tau_diffusion`` (from an earlier call with the same
    diffusion, law and trace space) skips its time-independent evaluation.
    """
    from hdgfem.hdg import condensation as hdg
    from hdgfem.assembly.advection_diffusion_reaction import (
        ADRPreparedData, _normal_flux, element_beta_samples, normalize_diffusion_stabilization)
    from hdgfem.hdg import matrices
    import hdgfem.hdg.stabilization as hdg_stabilization
    from hdgfem.hdg.reference import _reference_derivative_matrices
    from hdgfem.hdg.stabilization import is_conflict_averaged_upwind, upwind_factor
    from hdgfem.hdg.coefficients_device import volume_samples_cupy, face_samples_cupy
    from hdgfem.hdg.coefficients import beta_values_on_volume, reaction_values_on_volume

    cp = require_cupy()
    timings = {} if timings is None else timings
    total_start = start = time.perf_counter()
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    as_cupy_space(space)
    timings["raw.coefficients.space"] = _elapsed(cp, start)

    def host_beta():
        """Host beta.n and volume samples, exactly as ``prepare_adr_data``."""
        if isinstance(beta, ElementCoefficient):
            return element_beta_samples(beta, space, trace_ref, t=t)
        field = beta if isinstance(beta, VectorDGField) else hdg.as_vector_field(beta, space)
        return _normal_flux(beta, space, trace_ref), beta_values_on_volume(field, None, space)

    def host_scalar(value):
        """Resolve an element-local scalar to host quadrature values for a fallback."""
        return value.volume_values(space, t=t) if isinstance(value, ElementCoefficient) else value

    start = time.perf_counter()
    beta_dot_normal, beta_values = _host_fallback(
        cp, lambda: _beta_samples(cp, beta, space, trace_ref, t=t), host_beta, timings, "raw.coefficients.beta")
    timings["raw.coefficients.beta"] = _elapsed(cp, start)

    start = time.perf_counter()
    reaction_values = _host_fallback(
        cp, lambda: (reaction.volume_values(space, xp=cp, t=t) if isinstance(reaction, ElementCoefficient)
                     else volume_samples_cupy(reaction, space, label="reaction")),
        lambda: reaction_values_on_volume(host_scalar(reaction), space), timings, "raw.coefficients.reaction")
    timings["raw.coefficients.reaction"] = _elapsed(cp, start)

    start = time.perf_counter()
    source_rhs = _host_fallback(
        cp, lambda: _source_moments(cp, source, space, t=t),
        lambda: (hdg.source_moments_from_values(source.volume_values(space, t=t), space)
                 if isinstance(source, ElementCoefficient) else hdg.source_moments(source, space)),
        timings, "raw.coefficients.source")
    timings["raw.coefficients.source"] = _elapsed(cp, start)

    start = time.perf_counter()
    factor = upwind_factor(advection_stabilization)
    if factor is not None and not is_conflict_averaged_upwind(advection_stabilization):
        tau_advection = factor * cp.abs(beta_dot_normal)
    elif not is_conflict_averaged_upwind(advection_stabilization):
        tau_advection = _host_fallback(
            cp, lambda: face_samples_cupy(advection_stabilization, space,
                label="advection_stabilization", trace_space=trace_ref, t=t),
            lambda: hdg_stabilization.advection_trace_stabilization_values(
                space, cp.asnumpy(beta_dot_normal), advection_stabilization, trace_space=trace_ref),
            timings, "raw.coefficients.tau_advection")
    else:
        # Explicit tau forms and the conflict repair keep their host definitions.
        tau_advection = cp.asarray(hdg_stabilization.advection_trace_stabilization_values(
            space, cp.asnumpy(beta_dot_normal), advection_stabilization, trace_space=trace_ref))
    timings["raw.coefficients.tau_advection"] = _elapsed(cp, start)

    start = time.perf_counter()
    stabilization_kwargs = dict(diffusion=diffusion, penalty_constant=diffusion_penalty_constant,
                                trace_space=trace_ref, t=t)
    if tau_diffusion is None:
        tau_diffusion = _host_fallback(
            cp, lambda: normalize_diffusion_stabilization(diffusion_stabilization, space, device=True,
                                                          **stabilization_kwargs),
            lambda: normalize_diffusion_stabilization(diffusion_stabilization, space, **stabilization_kwargs),
            timings, "raw.coefficients.tau_diffusion")
    else:  # reused by a caller that owns the time-independent diffusion data
        tau_diffusion = cp.asarray(tau_diffusion)
        timings["raw.coefficients.tau_diffusion.cached"] = 1.0
    tau_samples = tau_diffusion[:, :, None] if tau_diffusion.ndim == 2 else tau_diffusion
    tau_total = cp.ascontiguousarray(tau_advection + tau_samples)
    gamma = cp.ascontiguousarray(tau_total - beta_dot_normal)
    timings["raw.coefficients.tau_diffusion"] = _elapsed(cp, start)

    d0_reference, d1_reference = _reference_derivative_matrices(space)
    spatial_law = callable(diffusion_stabilization) or hasattr(diffusion_stabilization, "space")
    preparation_seconds = time.perf_counter() - total_start
    return ADRPreparedData(
        source_rhs=source_rhs,
        reaction_values=cp.ascontiguousarray(reaction_values),
        beta_values=beta_values,
        beta_dot_normal=beta_dot_normal,
        tau_advection=cp.ascontiguousarray(tau_advection),
        tau_diffusion=tau_diffusion,
        tau_total=tau_total,
        gamma=gamma,
        u_boundary_mass=None, normal_mass_x=None, normal_mass_y=None,
        element_boundary=None, trace_lift=None, interior_gamma_mass=None,
        d0_reference=np.ascontiguousarray(d0_reference),
        d1_reference=np.ascontiguousarray(d1_reference),
        diffusion_stabilization_law=diffusion_stabilization if spatial_law else tau_diffusion,
        face_quadrature=trace_ref.quads.copy(), sample_time=t,
        preparation_seconds=preparation_seconds,
    )


__all__ = ["prepare_adr_data_cupy"]
