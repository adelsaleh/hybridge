"""Disk annulus reference from Zoni--Guclu (2019), section 6.3, eq. (34).

https://arxiv.org/abs/1909.05005
The closed dispersion relation is exact for a sharp constant-density annulus,
not for a smoothed/projected annulus. No temporal solver is used here.
"""
from __future__ import annotations

import math
import numpy as np

PAPER_PARAMETERS = dict(k=9, epsilon=1.e-4, s_minus=.45, s_plus=.50,
                        rho_bar=1., p=50., truncate=True)
PAPER_FIT_WINDOW = (20., 50.)


def annulus_spectrum(modes, *, inner=.45, outer=.50, wall=1., density=1.):
    """Return both frequencies, growth and discriminant of each integer mode.

    Convention: exp(i*(m*theta - omega*t)), omega_D=density/2.
    Positive imaginary frequency means growing amplitude, not growing energy.
    """
    if not all(math.isfinite(x) for x in (inner, outer, wall, density)):
        raise ValueError("annulus parameters must be finite")
    if not 0 <= inner < outer < wall or density <= 0:
        raise ValueError("require 0 <= inner < outer < wall and positive density")
    raw = np.asarray(tuple(modes))
    if raw.ndim != 1 or not np.all(np.isfinite(raw)) or np.any(raw < 1) or np.any(raw != np.floor(raw)):
        raise ValueError("modes must be positive integers")
    m = raw.astype(np.int64)
    lower, upper, ratio = inner/wall, outer/wall, inner/outer
    shear = 1-ratio**2
    b = m*shear + upper**(2*m) - lower**(2*m)
    c = m*shear*(1-lower**(2*m)) - (1-ratio**(2*m))*(1-upper**(2*m))
    discriminant = b*b-4*c
    root = np.sqrt(discriminant.astype(np.complex128))
    plus, minus = density*(b+root)/4, density*(b-root)/4
    return [dict(mode=int(k), omega_real=float(w.real), growth_rate=float(w.imag),
                 omega_other_real=float(v.real), omega_other_imag=float(v.imag),
                 unstable=bool(w.imag > 0), discriminant=float(d), b=float(bm), c=float(cm))
            for k,w,v,d,bm,cm in zip(m,plus,minus,discriminant,b,c)]


def fit_growth(times, amplitudes, *, window=PAPER_FIT_WINDOW, reference=None, minimum_samples=8):
    """Fit log amplitude on a predeclared complete window, with no point selection.

    Incomplete windows and nonpositive/nonfinite data cannot produce a pass.
    Standard errors describe the regression only, not discretization error.
    """
    t,a = np.asarray(times, dtype=float), np.asarray(amplitudes, dtype=float)
    start,end = map(float,window)
    if t.ndim != 1 or a.shape != t.shape or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError("finite strictly increasing times and matching amplitudes required")
    if not math.isfinite(start+end) or end <= start:
        raise ValueError("growth window must be finite and increasing")
    if int(minimum_samples)!=minimum_samples or minimum_samples<3:
        raise ValueError("at least three fit samples required")
    if len(t)<minimum_samples or t[0]>start+1.e-10 or t[-1]<end-1.e-10:
        return dict(status="incomplete_window", window=[start,end])
    mask = (t>=start-1.e-10)&(t<=end+1.e-10)
    x,y = t[mask],a[mask]
    if len(x)<minimum_samples:
        return dict(status="insufficient_samples", window=[start,end], samples=len(x))
    if not np.all(np.isfinite(y)) or np.any(y<=0):
        return dict(status="invalid_amplitude", window=[start,end], samples=len(x))
    y = np.log(y)
    centered = x-x.mean()
    variance = centered @ centered
    slope = float(centered @ (y-y.mean())/variance)
    intercept = float(y.mean()-slope*x.mean())
    residual = y-(intercept+slope*x)
    sse = float(residual @ residual)
    total = float((y-y.mean())@(y-y.mean()))
    out = dict(status="fitted", window=[start,end], samples=len(x), growth_rate=slope,
               log_intercept=intercept, r_squared=1-sse/total if total else None,
               slope_standard_error=math.sqrt(sse/(len(x)-2)/variance),
               amplitude_gain=float(math.exp(slope*(end-start))))
    if reference is not None:
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("growth reference must be positive")
        out.update(reference_growth_rate=reference, relative_error=(slope-reference)/reference)
    return out


def candidate_annulus(mode, *, center=.8):
    """Thin-annulus starting geometry with the selected mode near peak growth.

    m*(1-(r_in/r_out)**2) ~= 1.59 is an asymptotic design estimate. The exact
    finite-wall spectrum and actual mesh/projection checks must still be used.
    """
    if int(mode)!=mode or mode < 3 or not 0<center<1:
        raise ValueError("mode >= 3 and center inside the unit disk required")
    ratio=math.sqrt(1-1.59/mode)
    outer=2*center/(1+ratio)
    inner=ratio*outer
    if outer>=1:
        raise ValueError("candidate annulus intersects the wall")
    return dict(inner=inner,outer=outer,center=center,width=outer-inner)


def disk_poisson_green(mode, target, source, *, wall=1.):
    """Radial Green function for -Delta_m, regular at 0, Dirichlet at R.

    phi_m(r) = integral G_m(r,s) rho_m(s) s ds, for m >= 1.
    """
    r=np.asarray(target,dtype=float)[:,None]
    q=np.asarray(source,dtype=float)[None,:]
    ratio=np.divide(np.minimum(r,q),np.maximum(r,q),out=np.zeros((r.size,q.size)),where=np.maximum(r,q)>0)
    return (ratio**mode-(r*q/wall**2)**mode)/(2*mode)


def smooth_annulus_spectrum(modes, *, parameters, radial_points=64, wall=1.):
    """Numerical radial reference for the actual smooth equilibrium.

    Linearizing the same disk equations with exp(i*(m*theta-omega*t)) gives
      omega rho_m = m Omega rho_m + m rho0'(r)/r * phi_m,
      Omega(r) = r^-2 integral_0^r s rho0(s) ds.
    A Gauss Nystrom discretization of the analytic radial Poisson Green
    function gives a small dense eigenproblem. This is a reference calculation,
    independent of the HDG mesh and time integrator, not a closed-form rate.
    Reference convergence must be checked by increasing radial_points.
    """
    from scipy.integrate import quad
    from scipy.linalg import eigvals
    from scripts.guiding_center.cases.guiding_center_cases import diocotron_k

    case=diocotron_k(**parameters)
    if case.equilibrium_radial_derivative is None:
        raise ValueError("radial reference requires a smooth untruncated profile")
    inner,outer=case.parameters["s_minus"],case.parameters["s_plus"]
    center=case.parameters["s_bar"]
    if not 0<inner<center<outer<wall or radial_points<8:
        raise ValueError("require 0 < inner < center < outer < wall and at least 8 radial points")
    # Validate modes with the independent sharp-annulus reference API.
    checked=annulus_spectrum(modes,inner=inner,outer=outer,wall=wall,density=case.parameters["rho_bar"])
    points,weights=np.polynomial.legendre.leggauss(int(radial_points))
    edges=(0.,inner,center,outer,wall)
    radii=np.concatenate([(a+b)/2+(b-a)*points/2 for a,b in zip(edges,edges[1:])])
    weights=np.concatenate([weights*(b-a)/2 for a,b in zip(edges,edges[1:])])
    omega0=[]
    for r in radii:
        integral=quad(lambda s:float(case.equilibrium_density(s,0))*s,0,r,
                      points=[b for b in edges[1:-1] if b<r],epsabs=2.e-14,epsrel=2.e-12)[0]
        omega0.append(integral/r**2)
    omega0=np.asarray(omega0)
    derivative=case.equilibrium_radial_derivative(radii)/radii
    records=[]
    for entry in checked:
        mode=entry["mode"]
        matrix=mode*derivative[:,None]*disk_poisson_green(mode,radii,radii,wall=wall)*(radii*weights)[None,:]
        matrix[np.diag_indices_from(matrix)]+=mode*omega0
        eigenvalues=eigvals(matrix,overwrite_a=True,check_finite=False)
        fastest=eigenvalues[np.argmax(eigenvalues.imag)]
        growth=max(0.,float(fastest.imag))
        records.append(dict(mode=mode,growth_rate=growth,omega_real=float(fastest.real),
            unstable=bool(growth>1.e-10),reference_type="smooth-radial-nystrom",
            radial_points_per_segment=int(radial_points),radial_nodes=len(radii)))
    return records


def converged_smooth_reference(mode, *, parameters, resolutions=None):
    """Retain all refinements; thin high-mode rings need an extra radial level."""
    if resolutions is None:
        resolutions=(32,64,128,256) if mode>=32 else (32,64,128)
    if len(resolutions)<2 or any(b<=a for a,b in zip(resolutions,resolutions[1:])):
        raise ValueError("at least two increasing radial resolutions required")
    results=[smooth_annulus_spectrum([mode],parameters=parameters,radial_points=n)[0] for n in resolutions]
    last=results[-1]
    difference=abs(last["growth_rate"]-results[-2]["growth_rate"])
    relative=difference/max(last["growth_rate"],1.e-14)
    return dict(**last,refinements=results,refinement_absolute_change=difference,
                refinement_relative_change=relative,reference_converged=bool(relative<5.e-4))


def main():
    """Compute equation (34) and a smooth reference for an exact run preset."""
    from argparse import ArgumentParser
    import json
    from pathlib import Path
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.cases.guiding_center_cases import diocotron_k

    parser=ArgumentParser(description="Offline diocotron spectrum: Zoni–Güçlü equation (34) and smooth radial eigenvalues.")
    parser.add_argument("--preset",required=True)
    parser.add_argument("--max-mode",type=int)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=preset_by_key(args.preset)
    if config.case!="diocotron_k":
        parser.error("the reference requires a diocotron_k disk preset")
    case=diocotron_k(**config.case_params)
    parameters=case.parameters
    mode=int(parameters["k"])
    maximum=args.max_mode if args.max_mode is not None else max(32,2*mode)
    if maximum<mode:
        parser.error("max-mode must include the selected mode")
    sharp=annulus_spectrum(range(1,maximum+1),inner=parameters["s_minus"],
                           outer=parameters["s_plus"],density=parameters["rho_bar"])
    unstable=[r["mode"] for r in sharp if r["unstable"]]
    result=dict(source="https://arxiv.org/abs/1909.05005",section="6.3",equation=34,
        preset=args.preset,parameters=parameters,sharp_target=sharp[mode-1],
        sharp_spectrum=sharp,sharp_unstable_modes=unstable,
        sharp_fastest=max(sharp,key=lambda r:r["growth_rate"]),
        smooth_target=(converged_smooth_reference(mode,parameters=parameters)
                       if case.equilibrium_radial_derivative is not None else None),
        interpretation="The analytical formula applies to a sharp constant-density layer. The smooth reference is a separate numerical eigenvalue calculation.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Zoni–Güçlü sharp annulus: m={mode}, gamma={sharp[mode-1]['growth_rate']:.10f}, Re(omega)={sharp[mode-1]['omega_real']:.10f}")
    print(f"Unstable sharp-layer modes within 1..{maximum}: {unstable}")
    if result["smooth_target"] is not None:
        reference=result["smooth_target"]
        print(f"Smooth radial reference: gamma={reference['growth_rate']:.10f}, Re(omega)={reference['omega_real']:.10f}, converged={reference['reference_converged']}")
    print(args.output)


if __name__ == "__main__":
    main()
