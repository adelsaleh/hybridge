"""Connect preset policy and case boundaries to the guiding-center time steppers."""
from hdgfem.runtime.logging import timed_call
from scripts.guiding_center.time_schemes import STEPPERS, SIEulerStepper
from .configuration import (
    _detail_verbosity,
    _hybrid_startup_method,
    _phase_verbosity,
    _transport_trace_basis,
    _verbosity_level,
)


def make_stepper(config, case, space, density, poisson_result, density_trace,
                 potential_trace, *, transport_boundary_mode, poisson_solver,
                 positivity=None, recovery_record=None):
    """Initialize the selected algorithm using the tested preset's policies."""
    stepper_type = STEPPERS[config.time_scheme]
    options = dict(
        density_boundary=(lambda t: None) if transport_boundary_mode == "zero-flux" else case.density_boundary_at,
        potential_boundary=case.potential_boundary_at,
        poisson_solver=poisson_solver,
        poisson_tau_retry_factor=config.poisson_tau_retry_factor,
        poisson_tau_max_retries=config.poisson_tau_max_retries,
        recovery_verbosity=_verbosity_level(config), recovery_record=recovery_record,
        phase_verbosity=_phase_verbosity(config), detail_verbosity=_detail_verbosity(config),
    )
    if config.time_scheme == "si-bdf2":
        options["use_postprocessed_flux"] = config.transport_electric_field == "postprocessed"
    if issubclass(stepper_type, SIEulerStepper):
        return stepper_type(space, config.dt, density, poisson_result, density_trace,
                            potential_trace=potential_trace, **options)

    from hdgfem.transport.residual import HDGTraceWorkspace, UpwindHDGTransportResidual

    def initialize():
        """Prime the shared residual/projection workspace before normal timings."""
        startup_method = _hybrid_startup_method(config)
        needs_residual = config.time_scheme in {"h1-bdf3", "imex-ark3"} or startup_method == "ssprk3"
        workspace_type = UpwindHDGTransportResidual if needs_residual else HDGTraceWorkspace
        workspace = workspace_type(
            space, trace_basis=_transport_trace_basis(config),
            backend="device" if config.transport_assembly_backend in {"cupy", "raw-cuda"} else "host",
            **({"boundary_mode": transport_boundary_mode} if needs_residual else {}),
        )
        if config.time_scheme == "imex-ark3":
            options.update(
                density_diagnostics=None if positivity is None else positivity.measure,
            )
        else:
            options["startup_method"] = startup_method
        return stepper_type(space, config.dt, density, poisson_result, workspace, **options)

    stepper, elapsed = timed_call(
        f"[gc:init] priming {config.time_scheme.upper()} state and trace caches",
        _detail_verbosity(config), initialize,
    )
    stepper.setup_time = elapsed
    return stepper
