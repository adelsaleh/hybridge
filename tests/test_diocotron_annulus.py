"""Annulus diocotron benchmark: linear references, conforming mesh, case and diagnostics.

Linear-theory tables, mesh generation, projections and polar sampling only;
no linear PDE solve and no time integration.
"""
import numpy as np
import pytest

from hybridge.core.space import DGSpace
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.diagnostics.diocotron_reference import (
    SEMI_LAGRANGIAN_DIRICHLET_ANNULUS,
    SEMI_LAGRANGIAN_INNER_WALL,
    annular_poisson_green,
    annulus_spectrum,
    disk_poisson_green,
    interface_spectrum,
    smooth_annulus_spectrum,
)


@pytest.mark.parametrize("mode", [1, 2, 3, 9, 13, 14])
def test_interface_spectrum_without_inner_wall_is_zoni_guclu_equation_34(mode):
    closed = annulus_spectrum([mode])[0]
    interface = interface_spectrum([mode], inner=.45, outer=.50)[0]
    assert interface["growth_rate"] == pytest.approx(closed["growth_rate"], abs=1e-12)
    assert interface["omega_real"] == pytest.approx(closed["omega_real"], abs=1e-12)


@pytest.mark.parametrize("row", SEMI_LAGRANGIAN_DIRICHLET_ANNULUS, ids=lambda row: f"l{row['mode']}")
def test_inner_wall_spectrum_reproduces_madaule_table(row):
    reference = interface_spectrum([row["mode"]], inner=row["inner"], outer=row["outer"],
                                   inner_wall=SEMI_LAGRANGIAN_INNER_WALL)[0]
    # The table lists Im(omega/omega_D) = 2 * amplitude growth rate, rounded to 5 digits.
    assert 2 * reference["growth_rate"] == pytest.approx(row["analytic"], abs=1.5e-5)


def test_inner_wall_spectrum_is_scale_invariant():
    small = interface_spectrum([3], inner=.4, outer=.5, inner_wall=.1, wall=1.)[0]
    large = interface_spectrum([3], inner=4., outer=5., inner_wall=1., wall=10.)[0]
    assert small["growth_rate"] == pytest.approx(large["growth_rate"], rel=1e-10)


def test_annular_green_function_is_symmetric_vanishes_on_walls_and_reduces_to_disk():
    radii = np.linspace(.1, 1., 7)
    green = annular_poisson_green(3, radii, radii, inner_wall=.1, wall=1.)
    assert np.allclose(green, green.T)
    assert np.allclose(green[0], 0) and np.allclose(green[-1], 0)
    assert np.allclose(annular_poisson_green(4, radii, radii, inner_wall=0., wall=1.),
                       disk_poisson_green(4, radii, radii, wall=1.))
    # The m-th radial operator applied to G(., s) vanishes away from r = s.
    r = np.linspace(.3, .5, 2001)
    g = annular_poisson_green(2, r, [.8], inner_wall=.1, wall=1.)[:, 0]
    dr = r[1] - r[0]
    operator = -(np.gradient(r * np.gradient(g, dr), dr)) / r + 4 * g / r ** 2
    assert np.abs(operator[5:-5]).max() < 1e-4 * np.abs(g).max()


def test_smooth_inner_wall_reference_approaches_the_sharp_band():
    sharp = interface_spectrum([3], inner=.4, outer=.5, inner_wall=.1)[0]["growth_rate"]
    parameters = dict(k=3, epsilon=1e-6, s_minus=.4, s_plus=.5, p=32., truncate=False)
    smooth = smooth_annulus_spectrum([3], parameters=parameters, radial_points=64, inner_wall=.1)[0]
    assert smooth["growth_rate"] == pytest.approx(sharp, rel=.02)


def test_conforming_annulus_mesh_never_crosses_the_band_edges():
    pytest.importorskip("gmsh")
    from hybridge.core.mesh import gmsh_annulus_mesh

    mesh = gmsh_annulus_mesh(.08, inner_radius=.1, interface_radii=(.4, .5), band=(.4, .5),
                             band_mesh_size=.03, cache=False, log_cache=False)
    vertices = mesh.node_coords[mesh.triangles]
    vertex_radius = np.hypot(vertices[..., 0], vertices[..., 1])
    centroid_radius = np.hypot(*vertices.mean(axis=1).T)
    in_band = (centroid_radius > .4) & (centroid_radius < .5)
    assert in_band.any()
    assert not np.any(in_band[:, None] & ((vertex_radius < .4 - 1e-12) | (vertex_radius > .5 + 1e-12)))
    assert not np.any(~in_band[:, None] & ((vertex_radius > .4 + 1e-12) & (vertex_radius < .5 - 1e-12)))
    boundary = np.hypot(*mesh.node_coords[np.unique(mesh.edges[mesh.bnd_edges_inds])].T)
    assert np.allclose(np.sort(np.unique(np.round(boundary, 12))), [.1, 1.])
    edges = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2).max(axis=1)
    assert edges[in_band].max() < edges[~in_band].max()


def annulus_case(**params):
    return case_definition_by_key("diocotron_annulus").build(
        **{**dict(k=3, s_minus=.4, s_plus=.5, epsilon=1e-3), **params})


def test_annulus_case_geometry_and_profiles():
    case = annulus_case(band_mesh_size=.02)
    assert case.default_domain == "annulus"
    assert case.conforming_band == (.4, .5)
    assert case.parameters["geometry"]["interface_radii"] == (.4, .5)
    assert case.parameters["inner_wall"] == .1
    x = np.array([.45, .2, .7]); y = np.zeros(3)
    assert np.allclose(case.initial_density(x, y), [1.001, 0., 0.])
    assert np.allclose(case.band_modulation(x, y, False), 1.)
    smooth = annulus_case(p=8.)
    assert smooth.conforming_band is None and smooth.parameters["geometry"]["interface_radii"] == ()


def test_conforming_projection_is_the_exact_elementwise_band():
    pytest.importorskip("gmsh")
    from hybridge.core.mesh import gmsh_annulus_mesh
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.runner import _project_case_density

    case = annulus_case()
    mesh = gmsh_annulus_mesh(.08, inner_radius=.1, interface_radii=(.4, .5), cache=False, log_cache=False)
    space = DGSpace(mesh, 3, basis_type="dub_orth")
    config = preset_by_key("rho_helm_wave_host_accuracy")
    density = _project_case_density(config, space, case, equilibrium=False, name="rho")
    equilibrium = _project_case_density(config, space, case, equilibrium=True, name="rho_eq")
    centroid_radius = np.hypot(*mesh.node_coords[mesh.triangles].mean(axis=1).T)
    outside = (centroid_radius < .4) | (centroid_radius > .5)
    assert np.all(density.coeffs[outside] == 0.0)
    band_area = float(np.sum(mesh.aff_jacs[~outside]) * 2.0)  # reference triangle area is 2
    assert equilibrium.integral() == pytest.approx(band_area, rel=1e-12)
    modulation = space.project_callable(lambda x, y: case.band_modulation(x, y, True))
    assert np.allclose(density.coeffs[~outside], modulation.coeffs[~outside])


def test_polar_mode_sampling_covers_the_annulus_between_the_walls():
    pytest.importorskip("gmsh")
    from hybridge.core.mesh import gmsh_annulus_mesh
    from scripts.guiding_center.diagnostics.diocotron_diagnostics import DiocotronModeDiagnostics

    mesh = gmsh_annulus_mesh(.08, inner_radius=.1, interface_radii=(.4, .5), cache=False, log_cache=False)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    # A pure mode-3 potential perturbation on top of a radial "equilibrium".
    equilibrium = space.project_callable(lambda x, y: (1 - x * x - y * y))
    perturbed = space.project_callable(
        lambda x, y: (1 - x * x - y * y) + 1e-3 * (x ** 3 - 3 * x * y * y))
    diagnostics = DiocotronModeDiagnostics(space, equilibrium, mode=3, inner=.4, outer=.5,
                                           radial_points=16, inner_wall=.1)
    assert diagnostics.radii.min() > .1
    result = diagnostics.measure(perturbed)
    assert result["diocotron_dominant_mode_1"] == 3
    assert result["diocotron_phi_mode_3_l2"] > 100 * result["diocotron_phi_mode_1_l2"]


def test_truncated_profile_reference_converges_and_tends_to_the_sharp_band():
    from scripts.guiding_center.diagnostics.diocotron_reference import (
        PAPER_PARAMETERS, converged_truncated_reference, truncated_annulus_spectrum)

    sharp = interface_spectrum([9], inner=.45, outer=.5)[0]["growth_rate"]
    zoni_guclu = converged_truncated_reference(9, parameters=dict(PAPER_PARAMETERS))
    assert zoni_guclu["reference_converged"] and zoni_guclu["refinement_relative_change"] < 1e-6
    # The cut p=50 super-Gaussian grows 0.34% faster than the sharp band of eq. (34).
    assert zoni_guclu["growth_rate"] == pytest.approx(0.1802488, abs=1e-7)
    assert zoni_guclu["growth_rate"] / sharp - 1 == pytest.approx(3.44e-3, abs=2e-5)
    # p -> infinity recovers the sharp band at rate O(1/p).
    errors = [truncated_annulus_spectrum([9], parameters=dict(PAPER_PARAMETERS, p=power), radial_points=128)[0]
              ["growth_rate"] / sharp - 1 for power in (200., 1000.)]
    assert 0 < errors[1] < errors[0] < 1.5e-3 and errors[0] / errors[1] == pytest.approx(5., rel=.15)


def test_analysis_of_a_truncated_band_uses_the_truncated_reference():
    from scripts.guiding_center.cases.guiding_center_cases import diocotron_k
    from scripts.guiding_center.diagnostics.analyze_diocotron import analyze
    from scripts.guiding_center.diagnostics.diocotron_reference import PAPER_PARAMETERS

    parameters = diocotron_k(**PAPER_PARAMETERS).parameters
    gamma = 0.18024882
    rows = []
    for step, t in enumerate(np.arange(0, 70.01, 0.5)):
        amplitude = 1e-4 * np.exp(gamma * t)
        row = dict(step=step, time=t, mass=1., energy_from_q_l2=2., enstrophy=3., poisson_tau=4000.,
                   diocotron_phi_mode_9_l2=amplitude, diocotron_phi_eq_l2=amplitude,
                   diocotron_phi_harmonic_ratio=1e-3)
        rows.append(row)
    config = dict(case="diocotron_k", case_parameters=parameters, dt=.5, mesh_size=.008, order=6, triangles=10,
                  time_scheme="si-bdf2", poisson_tau_initial=4000.)
    rows[0]["run_configuration"] = config
    run = dict(path="synthetic.jsonl", sha256="synthetic", rows=rows, times=np.array([r["time"] for r in rows]),
               config=config, timings=[])
    growth = analyze(run)["growth"]
    assert growth["reference"]["reference_type"] == "truncated-profile-nystrom"
    assert growth["target_mode"]["relative_error"] == pytest.approx(0., abs=1e-6)
    assert growth["sharp_annulus_relative_error"] == pytest.approx(3.44e-3, abs=2e-5)


def test_gaussian_injection_injects_its_charge_with_a_unit_window():
    from scipy.integrate import quad

    from scripts.guiding_center.cases.guiding_center_cases import sin2_window

    rate, primitive = sin2_window(60.0, 5.0)
    assert quad(rate, 55.0, 70.0, points=[60.0, 65.0])[0] == pytest.approx(1.0, abs=1e-12)
    assert primitive(60.0) == 0.0 and primitive(62.5) == pytest.approx(0.5) and primitive(70.0) == 1.0
    case = annulus_case(k=9, s_minus=.45, s_plus=.5, inner_wall=0.0, p=4.0, epsilon=1e-4,
                        injection_charges=(0.1,), injection_width=0.06, injection_start=60.0, injection_duration=5.0)
    assert case.has_density_forcing and case.density_source_window == (60.0, 65.0)
    assert case.parameters["injection"]["total_charge"] == pytest.approx(0.1)
    x = np.linspace(-1, 1, 801)
    X, Y = np.meshgrid(x, x)
    inside = X ** 2 + Y ** 2 <= 1
    source = case.density_source(X, Y, 62.5)
    assert source[inside].sum() * (x[1] - x[0]) ** 2 == pytest.approx(0.1 * rate(62.5), rel=1e-9)
    assert np.allclose(case.density_source_profile(X, Y) * case.density_source_rate(62.5), source)
    assert case.density_source_mass(70.0) == pytest.approx(0.1)
    assert np.all(case.density_source(X, Y, 59.0) == 0.0)


def test_injectors_must_stay_away_from_the_walls():
    with pytest.raises(ValueError):
        annulus_case(injection_charges=(0.1,))  # centred source inside the inner wall


@pytest.mark.parametrize("n", [8, 10, 12])
def test_point_vortex_ring_window_matches_the_linearization(n):
    from scripts.guiding_center.diagnostics.diocotron_reference import (
        point_vortex_ring_growth, point_vortex_ring_window)

    low, high = point_vortex_ring_window(n)
    assert point_vortex_ring_growth(n, low - 0.02) > 1e-3
    assert point_vortex_ring_growth(n, low + 0.02) < 1e-6
    assert point_vortex_ring_growth(n, high - 0.1) < 1e-6
    assert point_vortex_ring_growth(n, high + 0.1) > 1e-3
    # Thomson: without a centre, eight or more vortices are unstable; seven are neutral.
    assert point_vortex_ring_growth(7, 0.0) < 1e-6 < point_vortex_ring_growth(9, 0.0)


def test_central_charge_slows_and_then_stabilizes_the_ring():
    params = dict(k=9, s_minus=.45, s_plus=.50, epsilon=1e-4, p=4.0, truncate=False)
    rates = [smooth_annulus_spectrum([9], parameters=params, radial_points=48, ring_edges=(1., 2., 3., 4., 6., 8.),
                                     central_charge=q)[0]["growth_rate"] for q in (0.0, 0.1, 0.3)]
    assert rates[0] == pytest.approx(0.17966, rel=2e-4)
    assert rates[1] == pytest.approx(0.12292, rel=2e-3)
    assert rates[2] < 1e-6
    with pytest.raises(ValueError):
        smooth_annulus_spectrum([9], parameters=params, radial_points=16, inner_wall=.1, central_charge=0.1)


def test_density_forcing_shifts_the_reaction_and_adds_the_scaled_source():
    pytest.importorskip("gmsh")
    from dataclasses import replace

    from hybridge.core.mesh import gmsh_annulus_mesh
    from scripts.guiding_center.runtime.forcing import DensityForcing

    mesh = gmsh_annulus_mesh(.1, inner_radius=0.0, interface_radii=(), cache=False, log_cache=False)
    space = DGSpace(mesh, 3, basis_type="dub_orth")
    case = annulus_case(k=9, s_minus=.45, s_plus=.5, inner_wall=0.0, p=4.0, epsilon=1e-4,
                        injection_charges=(0.2,), injection_width=0.1, injection_start=0.0, injection_duration=4.0)
    forcing = DensityForcing(space, replace(case, density_reaction=0.5))
    assert forcing.active
    one = space.constant(1.0)
    assert forcing.reaction(0.1, one).constant_value == pytest.approx(0.95)
    base = space.constant(1.0)
    staged = forcing.stage_source(base, 0.1, 2.0)
    injected = staged.integral() - base.integral()
    assert injected == pytest.approx(0.1 * 0.2 * 0.5, rel=1e-3)  # scale * Q * w(2) with w(2) = 2/T
    assert forcing.source(5.0) is None and forcing.stage_source(base, 0.1, 5.0) is base
    rhs = forcing.rhs(base, 5.0)  # only alpha*rho outside the window
    assert rhs.integral() == pytest.approx(0.5 * base.integral(), rel=1e-12)
    with pytest.raises(ValueError):
        forcing.check_scheme("predictor-corrector")


def test_smooth_switch_and_ionization_fields():
    from scipy.integrate import quad

    from scripts.guiding_center.cases.guiding_center_cases import smooth_switch

    rate, primitive = smooth_switch(70.0, 4.0)
    assert rate(70.0) == 0.0 and rate(72.0) == pytest.approx(0.5) and rate(74.0) == rate(90.0) == 1.0
    for t in (71.0, 74.0, 90.0):
        assert primitive(t) == pytest.approx(quad(rate, 60.0, t, points=[70.0, 74.0])[0], abs=1e-12)
    params = dict(k=9, s_minus=.45, s_plus=.5, inner_wall=0.0, p=4.0)
    uniform = annulus_case(**params, epsilon=0.0, ionization_rate=0.01)
    assert uniform.density_reaction == 0.01 and uniform.density_reaction_profile is None
    assert uniform.has_density_forcing and uniform.parameters["ionization"]["width"] is None
    x = np.array([0.45, 0.475, 0.5])
    assert np.allclose(uniform.exact_density(x, 0 * x, 30.0), np.exp(0.3) * uniform.equilibrium_density(x, 0 * x))
    column = annulus_case(**params, epsilon=0.0, ionization_rate=0.04, ionization_width=0.4)
    assert column.density_reaction == 0.0 and column.density_reaction_rate(1e-9) == 1.0
    nu = 0.04 * np.exp(-(x / 0.4) ** 2)
    assert np.allclose(column.exact_density(x, 0 * x, 10.0), np.exp(10.0 * nu) * column.equilibrium_density(x, 0 * x))
    # A seeded ring, or an off-centre cloud, has no exact density.
    assert annulus_case(**params, epsilon=1e-4, ionization_rate=0.04, ionization_width=0.4).exact_density is None
    puff = annulus_case(**params, epsilon=0.0, ionization_rate=0.05, ionization_width=0.1,
                        ionization_center=(0.475, 0.0), ionization_start=70.0, ionization_ramp=4.0)
    assert puff.exact_density is None and puff.density_reaction_rate(72.0) == pytest.approx(0.5)
    assert puff.density_reaction_profile(0.475, 0.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        annulus_case(**params, ionization_rate=0.05, ionization_width=0.0)


def _ionization_forcing():
    from hybridge.core.mesh import gmsh_annulus_mesh
    from scripts.guiding_center.runtime.forcing import DensityForcing

    mesh = gmsh_annulus_mesh(.15, inner_radius=0.0, interface_radii=(), cache=False, log_cache=False)
    space = DGSpace(mesh, 4, basis_type="dub_orth")
    case = annulus_case(k=9, s_minus=.45, s_plus=.5, inner_wall=0.0, p=4.0, epsilon=1e-2, ionization_rate=0.3,
                        ionization_width=0.2, ionization_center=(0.475, 0.0), ionization_start=1.0,
                        ionization_ramp=2.0)
    return space, case, DensityForcing(space, case), space.project_callable(case.initial_density, name="rho")


def test_space_dependent_reaction_enters_the_solve_and_the_mass_balance():
    pytest.importorskip("gmsh")
    space, case, forcing, rho = _ionization_forcing()
    assert forcing.active and forcing.has_reaction and forcing.time_dependent_reaction
    one = space.constant(1.0)
    assert forcing.reaction(0.1, one, 0.5) is one  # gas not yet switched on
    with pytest.raises(ValueError):
        forcing.reaction(0.1, one)
    reaction = forcing.reaction(0.1, one, 2.0)  # r(2) = 1/2
    assert np.allclose(reaction.coeffs, one.coeffs - 0.05 * forcing.nu().coeffs, atol=1e-14)
    # int nu_h rho_h equals the quadrature of nu * rho_h (nu_h is the L2 projection) ...
    points = space.mapped_quads()
    nu = case.density_reaction_profile(points[..., 0], points[..., 1])
    exact = np.einsum("K,Kq,q->", space.mesh.aff_jacs, nu * rho.values(), space.quad_data.Krf_w)
    assert forcing.production(rho, 4.0) == pytest.approx(exact, rel=1e-12)
    assert forcing.production(rho, 2.0) == pytest.approx(0.5 * exact, rel=1e-12)
    # ... and the explicit residual term of IMEX-ARK3 carries the same mass.
    assert forcing.rhs(rho, 4.0).integral() == pytest.approx(exact, rel=1e-12)
    assert forcing.rhs(rho, 0.5) is None


def test_space_dependent_reaction_device_terms_match_host():
    pytest.importorskip("gmsh")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    from scripts.guiding_center.runtime.forcing import DensityForcing
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space

    space, case, host, rho = _ionization_forcing()
    expected_production, expected_rhs = host.production(rho, 4.0), host.rhs(rho, 4.0).coeffs
    as_cupy_coefficients(rho, as_cupy_space(space, device=0))
    device = DensityForcing(space, case)
    assert device.production(rho, 4.0) == pytest.approx(expected_production, rel=1e-12)
    rhs = device.rhs(rho, 4.0)
    assert 0 in (rhs._device_coeffs or {})
    assert np.allclose(rhs.coeffs, expected_rhs, atol=1e-13)
    reaction = device.reaction(0.1, space.constant(1.0), 4.0, like=rho)
    assert 0 in (reaction._device_coeffs or {})
