"""Static projection and resolution audit; no Poisson solves or time integration."""
from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.diagnostics.guiding_center import (
    ScalarPositivityDiagnostics,
    guiding_center_field_diagnostics,
)
from hdgfem.diagnostics.errors import evaluate_scalar_error
from scripts.guiding_center.diagnostics.diocotron_reference import PAPER_PARAMETERS, annulus_spectrum, candidate_annulus
from scripts.guiding_center.cases.guiding_center_cases import diocotron_k
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.runner import _build_mesh, _project_initial_field


def audit_projection(*, mesh_size=.008, order=6, modes=(9,32,64,128), backend="device",
                     error_quadrature=24, volume_quad_1d=None, initial_projection_quad_1d=None, radial_power=50, truncate=True):
    """Project the paper profile and thinner high-mode candidates on one mesh.

    High-mode radii are only design candidates. Width/h and wavelength/h do
    not certify nonlinear resolution; measured projection error and subsequent
    h/p refinement are required. Radial power and support truncation are configurable.
    """
    config=replace(preset_by_key("diocotron_zg_m9_ark3_p6_h008_dt005_t70"),
                   mesh_size=mesh_size, order=order, verbosity=2,
                   volume_quad_1d=volume_quad_1d, initial_projection_quad_1d=initial_projection_quad_1d)
    if backend=="host":
        config=replace(config,poisson_assembly_backend="numpy",transport_assembly_backend="numpy")
    mesh=_build_mesh(config,diocotron_k(**PAPER_PARAMETERS))
    space=DGSpace(mesh,order,basis_type=config.basis,volume_quadrature=config.volume_quadrature,
                  volume_quad_1d=config.volume_quad_1d,edge_quad_1d=config.edge_quad_1d)
    check=ScalarPositivityDiagnostics(space,backend=backend)
    vertices=mesh.node_coords[mesh.triangles]
    centroid=vertices.mean(axis=1)
    center_r=np.linalg.norm(centroid,axis=1)
    cell_h=np.max(np.linalg.norm(vertices-np.roll(vertices,1,axis=1),axis=2),axis=1)
    records=[]
    zero=space.zeros()
    for mode in modes:
        parameters=dict(PAPER_PARAMETERS,k=int(mode),p=float(radial_power),truncate=bool(truncate))
        if mode!=9:
            ring=candidate_annulus(int(mode))
            parameters.update(s_minus=ring["inner"],s_plus=ring["outer"])
        case=diocotron_k(**parameters)
        field=_project_initial_field(config,space,case.initial_density,name="rho_preflight_h")
        metrics=check.measure(field)
        physics=guiding_center_field_diagnostics(field,zero,VectorDGField((zero,zero)),backend=backend)
        error=evaluate_scalar_error(field,case.initial_density,volume_quad_1d=error_quadrature,backend=backend)
        inner,outer=parameters["s_minus"],parameters["s_plus"]
        integration_bounds=(inner,outer) if truncate else (0.,1.)
        integration_points=None if truncate else [inner,(inner+outer)/2,outer]
        radial_integral=quad(lambda r: float(case.equilibrium_density(r,0))*r,*integration_bounds,points=integration_points,epsabs=1.e-13)[0]
        radial_l2=quad(lambda r: float(case.equilibrium_density(r,0))**2*r,*integration_bounds,points=integration_points,epsabs=1.e-13)[0]
        mass_exact=2*np.pi*radial_integral
        z_exact=np.pi*radial_l2*(1+parameters["epsilon"]**2/2)
        band=np.abs(center_r-(inner+outer)/2)<(outer-inner)/2+cell_h
        h95=float(np.quantile(cell_h[band],.95))
        spectrum=annulus_spectrum(range(1,max(32,2*mode)+1),inner=inner,outer=outer)
        record=dict(mode=mode,parameters=parameters,mesh_size=mesh_size,order=order,
            triangles=mesh.num_tri,backend=backend,
            projection_quad_1d=initial_projection_quad_1d,
            evolution_volume_points=len(space.quad_data.Krf_w),error_quad_1d=error_quadrature,
            band_h95=h95,annulus_width=outer-inner,cells_across_band=(outer-inner)/h95,
            tangential_wavelength=np.pi*(inner+outer)/mode,
            cells_per_wavelength=np.pi*(inner+outer)/mode/h95,
            radial_transition_scale=(outer-inner)/(2*parameters["p"]),
            nominal_subcell_spacing=h95/(order+1),
            projection_l2_error=error.metrics.l2,
            projection_relative_l2_error=error.metrics.l2/np.sqrt(2*z_exact),
            projection_linf_sampled_error=error.metrics.linf,
            mass=physics["mass"],analytic_mass=mass_exact,
            projection_mass_relative_error=(physics["mass"]-mass_exact)/mass_exact,
            enstrophy=.5*physics["rho_l2_squared"],analytic_enstrophy=z_exact,
            fastest_sharp_mode=max(spectrum,key=lambda r:r["growth_rate"])["mode"],
            reference=spectrum[mode-1], **metrics)
        records.append(record)
        print(f"m={mode:3d} width={outer-inner:.6f} cells/ring={record['cells_across_band']:.2f} "
              f"cells/wave={record['cells_per_wavelength']:.2f} relative L2={record['projection_relative_l2_error']:.3e} "
              f"min={metrics['rho_min_checked']:.3e} negative mass={metrics['rho_negative_mass_quadrature']:.3e}",flush=True)
        del field
    return records


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size",type=float,default=.008)
    parser.add_argument("--order",type=int,default=6)
    parser.add_argument("--modes",type=int,nargs="+",default=[9,32,64,128])
    parser.add_argument("--radial-power",type=float,default=50)
    parser.add_argument("--truncate",action=BooleanOptionalAction,default=True)
    parser.add_argument("--backend",choices=("host","device"),default="device")
    parser.add_argument("--error-quadrature",type=int,default=24)
    parser.add_argument("--volume-quad-1d",type=int)
    parser.add_argument("--initial-projection-quad-1d",type=int,default=32)
    parser.add_argument("--output",type=Path,default=Path("artifacts/diocotron_validation/projection.json"))
    args=parser.parse_args()
    records=audit_projection(mesh_size=args.mesh_size,order=args.order,modes=args.modes,backend=args.backend,
                             error_quadrature=args.error_quadrature,volume_quad_1d=args.volume_quad_1d,
                             initial_projection_quad_1d=args.initial_projection_quad_1d,radial_power=args.radial_power,truncate=args.truncate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(source="https://arxiv.org/abs/1909.05005",records=records),indent=2)+"\n")
    print(args.output)


if __name__=="__main__":
    main()
