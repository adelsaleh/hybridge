"""Analytical reference, mapped sampling, and host/device positivity checks."""
from dataclasses import replace
import json
import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.diagnostics import ScalarPositivityDiagnostics
from hdgfem.io.raster import RasterGeometry
from scripts.guiding_center.diagnostics.diocotron_reference import annulus_spectrum, fit_growth, candidate_annulus, PAPER_PARAMETERS
from scripts.guiding_center.diagnostics.diocotron_diagnostics import DiocotronModeDiagnostics
from scripts.guiding_center.cases.guiding_center_cases import diocotron_k


def small_space(order=2, basis="dub_orth"):
    return DGSpace(rectangle_mesh(2, 2, xlim=(-1, 1), ylim=(-1, 1)), order, basis_type=basis)


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return cp


def test_paper_spectrum_and_density_scaling():
    spectrum = annulus_spectrum(range(1, 31))
    assert [r["mode"] for r in spectrum if r["unstable"]] == list(range(2, 14))
    fastest = max(spectrum, key=lambda r: r["growth_rate"])
    assert fastest["mode"] == 9
    assert fastest["growth_rate"] == pytest.approx(.17963095941143925)
    assert fastest["omega_real"] == pytest.approx(.42750081053291755)
    doubled = annulus_spectrum([9], density=2)[0]
    assert doubled["growth_rate"] == pytest.approx(2*fastest["growth_rate"])
    assert doubled["omega_real"] == pytest.approx(2*fastest["omega_real"])
    assert all(not r["unstable"] for r in annulus_spectrum(range(1, 31), inner=0))


def test_high_mode_requires_thinner_ring():
    assert not annulus_spectrum([64])[0]["unstable"]
    ring = candidate_annulus(64)
    result = annulus_spectrum(range(1, 150), inner=ring["inner"], outer=ring["outer"])
    assert abs(max(result, key=lambda r:r["growth_rate"])["mode"]-64) <= 1
    assert ring["width"] < .011


@pytest.mark.parametrize("mode,gamma,last_unstable",[(64,.19864906225603912,102),(128,.1999249920521964,205)])
def test_high_mode_sharp_spectrum_matches_independent_interface_matrix(mode,gamma,last_unstable):
    from scripts.guiding_center.diagnostics.diocotron_reference import disk_poisson_green
    ring=candidate_annulus(mode)
    radii=np.array([ring["inner"],ring["outer"]])
    rotation=np.array([0.,.5*(1-(radii[0]/radii[1])**2)])
    sheet=mode*np.diag(rotation)+mode*np.array([1.,-1.])[:,None]/radii[:,None]*disk_poisson_green(mode,radii,radii)*radii[None,:]
    eigenvalues=np.linalg.eigvals(sheet)
    spectrum=annulus_spectrum(range(1,2*mode+1),inner=radii[0],outer=radii[1])
    assert max(eigenvalues.imag) == pytest.approx(gamma,abs=5.e-14)
    assert spectrum[mode-1]["growth_rate"] == pytest.approx(gamma,abs=5.e-14)
    assert [r["mode"] for r in spectrum if r["unstable"]] == list(range(2,last_unstable+1))


def test_growth_fits_amplitude_and_rejects_incomplete_or_invalid_windows():
    t = np.arange(0, 70.01, .5)
    gamma = annulus_spectrum([9])[0]["growth_rate"]
    amplitude = 2.e-8*np.exp(gamma*t)
    fit = fit_growth(t, amplitude, reference=gamma)
    assert fit["growth_rate"] == pytest.approx(gamma)
    assert fit["r_squared"] == pytest.approx(1.)
    assert fit_growth(t, amplitude**2)["growth_rate"] == pytest.approx(2*gamma)
    assert fit_growth(t[:50], amplitude[:50])["status"] == "incomplete_window"
    amplitude[60] = 0
    assert fit_growth(t, amplitude)["status"] == "invalid_amplitude"
    with pytest.raises(ValueError): fit_growth(t[::-1], amplitude[::-1])


def test_literal_paper_profile_has_compact_support_and_small_seed():
    case = diocotron_k(**PAPER_PARAMETERS)
    theta = np.arange(128)*2*np.pi/128
    x, y = .475*np.cos(theta), .475*np.sin(theta)
    np.testing.assert_allclose(case.initial_density(x,y), 1+1.e-4*np.cos(9*theta))
    x = np.array([.449, .451, .475, .499, .501])
    values = case.equilibrium_density(x, 0*x)
    assert values[0] == values[-1] == 0
    assert 0 < values[1] < values[2]
    assert case.parameters["truncate"] is True
    assert diocotron_k(p=50).parameters["truncate"] is False


@pytest.mark.parametrize("order,basis", [(0,"dub_orth"),(2,"dub_orth"),(6,"dub_orth"),(2,"bernstein")])
def test_polynomial_positivity_bounds_and_negative_mass(order, basis):
    s=small_space(order,basis)
    check=ScalarPositivityDiagnostics(s,chunk_size=1)
    positive=check.measure(s.constant(1))
    assert positive["positivity_status"] == "bound_satisfied"
    negative=check.measure(s.constant(-.25))
    assert negative["positivity_status"] == "violated"
    assert negative["rho_negative_mass_quadrature"] == pytest.approx(1.)
    assert negative["rho_negative_l2_quadrature"] == pytest.approx(.5)
    assert negative["rho_cell_average_min"] == pytest.approx(-.25)
    assert negative["rho_negative_cells_sampled"] == s.mesh.num_tri


def test_negative_bound_does_not_claim_negative_polynomial():
    s=DGSpace(rectangle_mesh(1,1,xlim=(-1,1),ylim=(-1,1)),2,basis_type="dub_orth")
    values=ScalarPositivityDiagnostics(s).measure(s.project_callable(lambda x,y:x*x+y*y+.01))
    assert values["rho_bernstein_lower_bound"] < 0
    assert values["rho_min_checked"] > 0
    assert values["positivity_status"] == "inconclusive"


def test_mapped_sampling_and_modal_parseval_norm_and_phase():
    s=small_space()
    theta=.3
    f=s.project_callable(lambda x,y: (x*x-y*y)*np.cos(theta)+2*x*y*np.sin(theta)+.2)
    diagnostics=DiocotronModeDiagnostics(s,s.zeros(),mode=2)
    result=diagnostics.measure(f)
    assert result["diocotron_phi_mode_target_l2"] == pytest.approx(np.sqrt(np.pi/6),rel=1.e-10)
    assert result["diocotron_phi_axisymmetric_l2"] == pytest.approx(.2*np.sqrt(np.pi),rel=1.e-10)
    assert result["diocotron_phi_nonaxisymmetric_l2"] == pytest.approx(np.sqrt(np.pi/6),rel=1.e-10)
    assert result["diocotron_phi_mode_phase"] == pytest.approx(-theta)
    assert result["diocotron_phi_harmonic_ratio"] < 1.e-12
    with pytest.raises(ValueError,match="six|6"):
        DiocotronModeDiagnostics(s,s.zeros(),mode=9,angular_points=32)
    points=np.array([[.2,.3],[2,0]])
    geometry=RasterGeometry.from_points(s.mesh,points,width=2,height=1)
    np.testing.assert_array_equal(geometry.valid_pixels,[0])
    assert (geometry.sampling_matrix(s) @ s.project_callable(lambda x,y:x+2*y).coeffs.ravel())[0] == pytest.approx(.8)


def test_high_mode_diagnostics_retain_neighboring_modes():
    # A low-order polynomial supplies an exact signal in the same cached FFT
    # used for the high-mode band. Neighboring high modes must remain observable.
    space=small_space(2)
    diagnostics=DiocotronModeDiagnostics(space,space.zeros(),mode=64,radial_points=4)
    result=diagnostics.measure(space.project_callable(lambda x,y:x*x-y*y))
    assert diagnostics.angular_points == 512
    assert all(f"diocotron_phi_mode_{m}_l2" in result for m in range(1,193))
    assert result["diocotron_phi_mode_2_l2"] == pytest.approx(np.sqrt(np.pi/6),rel=1.e-10)
    assert result["diocotron_phi_mode_63_l2"] < 1.e-12
    assert result["diocotron_phi_mode_65_l2"] < 1.e-12


def test_device_diagnostics_match_host_without_materializing_coefficients(cp):
    from hdgfem.backends.cupy import field_from_cupy_coefficients
    s=small_space(6)
    host=s.project_callable(lambda x,y:x*x-y*y+.1)
    device=field_from_cupy_coefficients(s,cp.asarray(host.coeffs).copy())
    zero=field_from_cupy_coefficients(s,cp.zeros(s.shape))
    for host_values, device_values in (
        (ScalarPositivityDiagnostics(s).measure(host),ScalarPositivityDiagnostics(s,backend="device").measure(device)),
        (DiocotronModeDiagnostics(s,s.zeros(),mode=2).measure(host),
         DiocotronModeDiagnostics(s,zero,mode=2,backend="device").measure(device)),
    ):
        for key,value in host_values.items():
            if isinstance(value,(int,float)):
                assert device_values[key] == pytest.approx(value,rel=1.e-10,abs=1.e-11)
    assert not device.coefficients_materialized and not zero.coefficients_materialized


def test_richer_projection_reuses_rule_and_matches_host_on_device(cp):
    from hdgfem.assembly.projection import project_callable
    from hdgfem.backends.cupy import as_cupy_space, as_cupy_coefficients
    s=small_space(3)
    function=lambda x,y:(x**6+2*y**4)*(x<.3)
    host=project_callable(function,s,volume_quad_1d=12)
    device=project_callable(function,s,backend="device",volume_quad_1d=12)
    cspace=as_cupy_space(s)
    operator=next(iter(cspace._callable_projection_cache.values()))[-1]
    again=project_callable(function,s,backend="device",volume_quad_1d=12)
    assert operator is next(iter(cspace._callable_projection_cache.values()))[-1]
    np.testing.assert_allclose(as_cupy_coefficients(device,cspace).get(),host.coeffs,rtol=2.e-12,atol=1.e-13)
    np.testing.assert_allclose(as_cupy_coefficients(again,cspace).get(),host.coeffs,rtol=2.e-12,atol=1.e-13)
    assert not device.coefficients_materialized
    # An exactly representable polynomial remains exact after overintegration.
    exact=project_callable(lambda x,y:1+x-2*y+x*y,s,backend="device",volume_quad_1d=12)
    np.testing.assert_allclose(as_cupy_coefficients(exact,cspace).get(),s.project_callable(lambda x,y:1+x-2*y+x*y).coeffs,atol=2.e-13)


def test_ark_positivity_checks_all_stages_and_distinct_endpoint():
    from test_guiding_center_imex_ark3 import scalar_stepper
    stepper,poisson,transport,calls,scalar=scalar_stepper(.1)
    check=ScalarPositivityDiagnostics(stepper.space)
    stepper.density_diagnostics=check.measure
    result=stepper.advance(poisson,transport)
    entries=result.metrics["positivity_stage_checks"]
    assert [r["stage"] for r in entries] == ["ARK stage 2","ARK stage 3","ARK stage 4","accepted endpoint"]
    np.testing.assert_allclose([r["rho_min_checked"] for r in entries[:3]], [c["value"] for c in calls])
    assert entries[-1]["rho_min_checked"] == pytest.approx(scalar(result.density))
    assert abs(entries[-1]["rho_min_checked"]-entries[-2]["rho_min_checked"]) > 1.e-7
    assert result.metrics["positivity_stage_time"] > 0


def test_paper_cli_enables_stage_and_modal_checks(monkeypatch,capsys):
    from scripts.guiding_center.runtime import runner
    import scripts.guiding_center.run_guiding_center_cases as cli
    monkeypatch.setattr("sys.argv",["gc","--preset","diocotron_zg_m9_ark3_p6_h008_dt005_t70","--dry-run"])
    cli._main()
    output=capsys.readouterr().out
    assert "positivity_diagnostics: True" in output
    assert "diocotron_diagnostics: True" in output
    assert "initial_projection_quad_1d: 32" in output


def test_radial_green_function_and_sharp_sheet_limit():
    from scripts.guiding_center.diagnostics.diocotron_reference import disk_poisson_green
    from scipy.integrate import quad
    for m in (2,9,20):
        for r in (.1,.47,.8):
            measured=quad(lambda s: disk_poisson_green(m,[r],[s])[0,0]*s**(m+1),0,1,points=[r],epsabs=1.e-13)[0]
            assert measured == pytest.approx((r**m-r**(m+2))/(4*(m+1)),abs=1.e-13)
        radii=np.array([.45,.50])
        omega0=np.array([0.,.5*(1-(radii[0]/radii[1])**2)])
        sheet=m*np.diag(omega0)+m*np.array([1.,-1.])[:,None]/radii[:,None]*disk_poisson_green(m,radii,radii)*radii[None,:]
        eigen=np.linalg.eigvals(sheet).astype(complex)
        reference=annulus_spectrum([m])[0]
        assert max(eigen.imag) == pytest.approx(reference["growth_rate"],abs=3.e-14)
        assert sum(eigen.real) == pytest.approx(reference["omega_real"]+reference["omega_other_real"])


def test_smooth_derivative_and_radial_reference_refinement():
    from scripts.guiding_center.diagnostics.diocotron_reference import smooth_annulus_spectrum
    parameters={**PAPER_PARAMETERS,"p":4.,"truncate":False}
    case=diocotron_k(**parameters)
    r=np.array([.442,.453,.478,.493,.519])
    delta=1.e-7
    finite=(case.equilibrium_density(r+delta,0*r)-case.equilibrium_density(r-delta,0*r))/(2*delta)
    np.testing.assert_allclose(case.equilibrium_radial_derivative(r),finite,rtol=3.e-8,atol=1.e-8)
    values=[smooth_annulus_spectrum([9],parameters=parameters,radial_points=n)[0] for n in (32,64,128)]
    assert values[-1]["growth_rate"] == pytest.approx(.17965,rel=2.e-4)
    assert abs(values[-1]["growth_rate"]-values[-2]["growth_rate"]) < .3*abs(values[-2]["growth_rate"]-values[-3]["growth_rate"])
    assert values[-1]["omega_real"] == pytest.approx(.3930597,rel=1.e-5)
    with pytest.raises(ValueError,match="smooth"):
        smooth_annulus_spectrum([9],parameters=PAPER_PARAMETERS)


def synthetic_run(epsilon=1.e-4,dt=1.,negative_stage=False):
    """Prescribed exponential samples, not numerical time integration."""
    from scripts.guiding_center.cases.guiding_center_cases import diocotron_k
    parameters=diocotron_k(**{**PAPER_PARAMETERS,"p":None,"truncate":False,"epsilon":epsilon}).parameters
    gamma=annulus_spectrum([9])[0]["growth_rate"]
    config=dict(case="diocotron_k",case_parameters=parameters,dt=dt,mesh_size=.008,order=6,
                triangles=10,time_scheme="imex-ark3",poisson_tau_initial=4000.)
    check=dict(rho_min_checked=.1,rho_cell_average_min=.1,rho_bernstein_lower_bound=.09,
               rho_negative_mass_quadrature=0.,positivity_tolerance=1.e-12,positivity_status="bound_satisfied")
    rows=[];timings=[]
    for step,t in enumerate(np.arange(0,70.01,dt)):
        amplitude=(epsilon+1.e-10)*np.exp(gamma*t)
        row=dict(step=step,time=t,mass=1.,energy_from_q_l2=2.,enstrophy=3.,poisson_tau=4000.,
                 diocotron_phi_mode_9_l2=amplitude,diocotron_phi_mode_target_l2=amplitude,
                 diocotron_phi_eq_l2=amplitude,diocotron_phi_harmonic_ratio=.001,**check)
        if step==0: row["run_configuration"]=config
        rows.append(row)
        if step:
            checks=[dict(time=t,stage=f"ARK stage {n}",**check) for n in (2,3,4)]
            if negative_stage and step==1: checks[0].update(rho_min_checked=-.1,positivity_status="violated")
            timings.append(dict(step=step,time=t,poisson_tau=4000.,positivity_stage_checks=checks))
        else: timings.append(row)
    return dict(path="synthetic.jsonl",sha256="synthetic",rows=rows,times=np.array([r["time"] for r in rows]),config=config,timings=timings)


def test_analysis_requires_controls_and_catches_stage_only_negativity():
    from scripts.guiding_center.diagnostics.analyze_diocotron import analyze
    main=synthetic_run(negative_stage=True)
    summary=analyze(main,control=synthetic_run(epsilon=0),half_seed=synthetic_run(epsilon=5.e-5),refined=synthetic_run(dt=.5))
    assert summary["growth"]["agrees_with_reference"]
    assert summary["growth"]["control"]["max_control_fraction"] < 1.e-5
    assert summary["positivity"]["status"] == "violated"
    assert summary["positivity"]["first_negative_witness"]["stage"] == "ARK stage 2"
    assert summary["invariants"]["enstrophy"]["final_relative_drift"] == 0
    assert not summary["qualification_notes"]
    missing=analyze(main)
    assert any("unperturbed" in message for message in missing["qualification_notes"])
    assert any("smaller timestep" in message for message in missing["qualification_notes"])
    mismatched=synthetic_run(epsilon=0)
    mismatched["config"]["order"]=4
    with pytest.raises(ValueError,match="configuration"):
        analyze(main,control=mismatched)


def test_analysis_marks_tau_changes_and_never_fits_short_runs():
    from scripts.guiding_center.diagnostics.analyze_diocotron import analyze
    short=synthetic_run()
    short["rows"]=short["rows"][:10];short["times"]=short["times"][:10];short["timings"]=short["timings"][:10]
    short["timings"][2]["poisson_tau_retry_events"]=[dict(tau_before=4000,tau_after=8000)]
    result=analyze(short)
    assert result["growth"]["target_mode"]["status"] == "incomplete_window"
    assert not result["growth"]["agrees_with_reference"]
    assert not result["poisson_tau"]["fixed"]
    assert any("tau changed" in note for note in result["qualification_notes"])


def test_high_mode_competitor_is_flagged_even_with_weak_second_harmonic():
    from scripts.guiding_center.diagnostics.analyze_diocotron import analyze
    run=synthetic_run()
    for row in run["rows"]:
        row["diocotron_phi_mode_17_l2"]=.2*row["diocotron_phi_mode_9_l2"]
    result=analyze(run)
    competitor=result["growth"]["maximum_competing_mode_fraction"]
    assert competitor["mode"]==17
    assert competitor["ratio"]==pytest.approx(.2)
    assert any("nonharmonic" in note for note in result["qualification_notes"])
