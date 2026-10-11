"""Linear diocotron references for the disk and the annulus with an inner wall.

* Zoni--Guclu (2019), section 6.3, eq. (34): sharp band in the unit disk
  (https://arxiv.org/abs/1909.05005), :func:`annulus_spectrum`.
* :func:`interface_spectrum`: sharp band between an optional inner wall and the
  outer wall, phi = 0 on both ("Dirichlet for all modes"). It is the 2x2
  surface-charge eigenproblem of the linearized equations with the annular
  Dirichlet Green function; it reproduces eq. (34) without inner wall and the
  analytic values of Madaule et al. (2013), HAL hal-00841504, Table 1.
* :func:`smooth_annulus_spectrum`: Nystrom radial eigenproblem for smooth
  profiles, with or without inner wall.

Convention: exp(i*(m*theta - omega*t)); ``growth_rate`` is Im(omega), the
growth rate of the mode amplitude. Semi-Lagrangian tables report
Im(omega/omega_D) with omega_D = 1/2, i.e. twice our amplitude rate.
No temporal solver is used here.
"""
from __future__ import annotations

import math
import numpy as np

PAPER_PARAMETERS = dict(k=9, epsilon=1.e-4, s_minus=.45, s_plus=.50,
                        rho_bar=1., p=50., truncate=True)
PAPER_FIT_WINDOW = (20., 50.)

# Madaule, Hirstoaga, Mehrenberger, Petri (2013), HAL hal-00841504, Table 1,
# "Dirichlet for all modes": inner wall r_min=1, outer wall r_max=10, rho=1 on
# [r_minus, r_plus], epsilon*cos(l*theta). Values are Im(omega/omega_D), i.e.
# 2*gamma (the growth rate of the squared mode amplitude). Radii are scaled
# here to an outer wall of radius 1; times are scale invariant.
SEMI_LAGRANGIAN_DIRICHLET_ANNULUS = (
    dict(mode=2, inner=.4, outer=.5, analytic=.28874, semi_lagrangian=.2875, fit_window=(26., 55.)),
    dict(mode=3, inner=.4, outer=.5, analytic=.36732, semi_lagrangian=.3667, fit_window=(26., 72.)),
    dict(mode=4, inner=.4, outer=.5, analytic=.38408, semi_lagrangian=.3852, fit_window=(26., 51.)),
    dict(mode=5, inner=.7, outer=.8, analytic=.32358, semi_lagrangian=.3222, fit_window=(22., 62.)),
    dict(mode=7, inner=.6, outer=.7, analytic=.33757, semi_lagrangian=.3424, fit_window=(26., 61.)),
)
SEMI_LAGRANGIAN_INNER_WALL = .1


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


def annular_poisson_green(mode, target, source, *, inner_wall=0., wall=1.):
    """Radial Green function for -Delta_m with phi=0 at r=inner_wall and r=wall.

    phi_m(r) = integral G_m(r,s) rho_m(s) s ds, m >= 1. With
    u1 = (r/a)^m-(a/r)^m and u2 = (r/b)^m-(b/r)^m vanishing at a=inner_wall
    and b=wall, G = -u1(r<) u2(r>) / (2m((b/a)^m-(a/b)^m)). ``inner_wall=0``
    reduces to :func:`disk_poisson_green`.
    """
    if inner_wall <= 0:
        return disk_poisson_green(mode, target, source, wall=wall)
    r=np.asarray(target,dtype=float)[:,None]
    q=np.asarray(source,dtype=float)[None,:]
    lower,upper=np.minimum(r,q),np.maximum(r,q)
    u1=(lower/inner_wall)**mode-(inner_wall/lower)**mode
    u2=(upper/wall)**mode-(wall/upper)**mode
    return -u1*u2/(2*mode*((wall/inner_wall)**mode-(inner_wall/wall)**mode))


def _enclosed_charge(radius, density, *, inner_wall, breaks):
    """M(r) = integral_{inner_wall}^r s rho0(s) ds for a radial density."""
    from scipy.integrate import quad

    if radius <= inner_wall:
        return 0.
    return quad(lambda s:float(density(s))*s,inner_wall,radius,
                points=[b for b in breaks if inner_wall<b<radius],epsabs=2.e-14,epsrel=2.e-12,limit=200)[0]


def _inner_wall_flux(density, *, inner_wall, wall, breaks):
    """C = inner_wall*phi0'(inner_wall) fixing phi0(wall)=0, so Omega=(M-C)/r^2."""
    from scipy.integrate import quad

    if inner_wall <= 0:
        return 0.
    integral=quad(lambda s:_enclosed_charge(s,density,inner_wall=inner_wall,breaks=breaks)/s,
                  inner_wall,wall,points=[b for b in breaks if inner_wall<b<wall],
                  epsabs=2.e-14,epsrel=2.e-12,limit=200)[0]
    return integral/math.log(wall/inner_wall)


def interface_spectrum(modes, *, inner, outer, wall=1., inner_wall=0., density=1.):
    """Sharp-band spectrum with an optional inner wall (phi=0 on both walls).

    The step rho0 = density on [inner, outer] makes rho_1 a pair of surface
    charges at the band edges; the linearized equations
    omega rho_1 = m Omega rho_1 + m rho0'/r phi_1 then reduce to a 2x2
    eigenproblem built from :func:`annular_poisson_green`. The angular
    velocity Omega = (M(r)-C)/r^2 includes the inner-wall charge.
    """
    if not 0 <= inner_wall < inner < outer < wall or density <= 0:
        raise ValueError("require 0 <= inner_wall < inner < outer < wall and positive density")
    raw=np.asarray(tuple(modes))
    if raw.ndim != 1 or np.any(raw < 1) or np.any(raw != np.floor(raw)):
        raise ValueError("modes must be positive integers")

    def enclosed(r):
        low,high=max(inner,inner_wall),min(r,outer)
        return 0. if high<=low else .5*density*(high*high-low*low)

    if inner_wall > 0:
        # Exact integral of M(s)/s for the step profile (M = 0 below the band).
        band=.5*density*((outer*outer-inner*inner)/2-inner*inner*math.log(outer/inner))
        beyond=.5*density*(outer*outer-inner*inner)*math.log(wall/outer)
        flux=(band+beyond)/math.log(wall/inner_wall)
    else:
        flux=0.
    omega_inner=(enclosed(inner)-flux)/inner**2
    omega_outer=(enclosed(outer)-flux)/outer**2
    records=[]
    for mode in raw.astype(np.int64):
        m=int(mode)
        green=annular_poisson_green(m,(inner,outer),(inner,outer),inner_wall=inner_wall,wall=wall)
        matrix=np.array([[m*omega_inner+m*density*green[0,0], m*density*green[0,1]*outer/inner],
                         [-m*density*green[1,0]*inner/outer, m*omega_outer-m*density*green[1,1]]])
        eigenvalues=np.linalg.eigvals(matrix)
        fastest,other=sorted(eigenvalues,key=lambda value:(round(value.imag,14),value.real),reverse=True)
        records.append(dict(mode=m,omega_real=float(fastest.real),growth_rate=float(fastest.imag),
            omega_other_real=float(other.real),omega_other_imag=float(other.imag),
            unstable=bool(fastest.imag>1.e-12),inner_wall=float(inner_wall),
            reference_type="sharp-band-interface"))
    return records


def smooth_annulus_spectrum(modes, *, parameters, radial_points=64, wall=1., inner_wall=0., ring_edges=None,
                            central_charge=0.):
    """Numerical radial reference for the actual smooth equilibrium.

    Linearizing the same disk equations with exp(i*(m*theta-omega*t)) gives
      omega rho_m = m Omega rho_m + m rho0'(r)/r * phi_m,
      Omega(r) = r^-2 integral_0^r s rho0(s) ds.
    A Gauss Nystrom discretization of the analytic radial Poisson Green
    function gives a small dense eigenproblem. This is a reference calculation,
    independent of the HDG mesh and time integrator, not a closed-form rate.
    Reference convergence must be checked by increasing radial_points.
    ``inner_wall > 0`` adds a conducting inner wall (phi=0): the Green function
    is :func:`annular_poisson_green` and Omega = (M(r)-C)/r^2 includes the
    inner-wall charge C fixed by phi0(wall)=0. ``ring_edges`` (multiples of the
    profile width s_d) adds Gauss segment breaks at s_bar +- w s_d, which
    resolves thin rings and the Green-function kink far better than the
    default five breaks. ``central_charge`` Q adds a charge concentrated at the
    centre of the disk (an injected central vortex): it rotates the ring with
    Q/(2 pi r^2) and, being far inside it, does not couple to its modes.
    """
    from scipy.linalg import eigvals
    from scripts.guiding_center.cases.guiding_center_cases import diocotron_k

    import inspect

    accepted=inspect.signature(diocotron_k).parameters
    case_parameters={key:value for key,value in parameters.items() if key in accepted}
    case=diocotron_k(**case_parameters)
    if case.equilibrium_radial_derivative is None:
        raise ValueError("radial reference requires a smooth untruncated profile")
    inner,outer=case.parameters["s_minus"],case.parameters["s_plus"]
    center=case.parameters["s_bar"]
    if not 0<=inner_wall<inner<center<outer<wall or radial_points<8:
        raise ValueError("require 0 <= inner_wall < inner < center < outer < wall and at least 8 radial points")
    # Validate modes with the independent sharp-band reference API.
    checked=interface_spectrum(modes,inner=inner,outer=outer,wall=wall,inner_wall=inner_wall,
                               density=case.parameters["rho_bar"])
    points,weights=np.polynomial.legendre.leggauss(int(radial_points))
    edges=(float(inner_wall),inner,center,outer,wall)
    if ring_edges:
        width=float(case.parameters["s_d"])
        extra={center+sign*float(w)*width for w in ring_edges for sign in (-1.,1.)}
        edges=tuple(sorted({e for e in (*edges,*extra) if float(inner_wall)<=e<=wall}))
    radii=np.concatenate([(a+b)/2+(b-a)*points/2 for a,b in zip(edges,edges[1:])])
    weights=np.concatenate([weights*(b-a)/2 for a,b in zip(edges,edges[1:])])
    density=lambda s:case.equilibrium_density(s,0)
    flux=_inner_wall_flux(density,inner_wall=float(inner_wall),wall=wall,breaks=edges[1:-1])
    if central_charge and inner_wall > 0:
        raise ValueError("a central charge needs the disk (inner_wall = 0)")
    omega0=np.asarray([(_enclosed_charge(r,density,inner_wall=float(inner_wall),breaks=edges[1:-1])-flux)/r**2
                       for r in radii])+float(central_charge)/(2*np.pi*radii**2)
    derivative=case.equilibrium_radial_derivative(radii)/radii
    records=[]
    for entry in checked:
        mode=entry["mode"]
        green=annular_poisson_green(mode,radii,radii,inner_wall=float(inner_wall),wall=wall)
        matrix=mode*derivative[:,None]*green*(radii*weights)[None,:]
        matrix[np.diag_indices_from(matrix)]+=mode*omega0
        eigenvalues=eigvals(matrix,overwrite_a=True,check_finite=False)
        fastest=eigenvalues[np.argmax(eigenvalues.imag)]
        growth=max(0.,float(fastest.imag))
        records.append(dict(mode=mode,growth_rate=growth,omega_real=float(fastest.real),
            unstable=bool(growth>1.e-10),reference_type="smooth-radial-nystrom",
            radial_points_per_segment=int(radial_points),radial_nodes=len(radii),
            ring_edges=None if not ring_edges else [float(w) for w in ring_edges]))
    return records


_SMOOTH_REFERENCE_CACHE: dict[str, dict] = {}
# Breaks at s_bar +- w s_d resolve thin rings; the Green-function kink makes Gauss-Nystrom
# second order, so the reference is Richardson-extrapolated from three refinements.
RING_EDGES = (1., 2., 3., 4., 6., 8.)


def converged_smooth_reference(mode, *, parameters, resolutions=(64, 128, 256), wall=1., inner_wall=0.,
                               ring_edges=RING_EDGES, central_charge=0.):
    """Ring-refined Nystrom rate, Richardson-extrapolated from three resolutions (memoized).

    ``growth_rate`` is the extrapolated value; ``refinements`` keeps the raw rates,
    ``richardson_order`` the observed order (about 2), and
    ``refinement_relative_change`` the size of the extrapolation correction.
    """
    import json

    key = json.dumps(dict(mode=int(mode), parameters=parameters, resolutions=resolutions, wall=float(wall),
                          inner_wall=float(inner_wall), ring_edges=ring_edges,
                          central_charge=float(central_charge)), sort_keys=True, default=str)
    cached = _SMOOTH_REFERENCE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    result = _converged_smooth_reference(mode, parameters=parameters, resolutions=resolutions,
                                         wall=wall, inner_wall=inner_wall, ring_edges=ring_edges,
                                         central_charge=central_charge)
    _SMOOTH_REFERENCE_CACHE[key] = dict(result)
    return result


def _converged_smooth_reference(mode, *, parameters, resolutions=(64, 128, 256), wall=1., inner_wall=0.,
                                ring_edges=RING_EDGES, central_charge=0.):
    """Uncached :func:`converged_smooth_reference`."""
    if len(resolutions) < 3 or any(b != 2*a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError("three or more resolutions, each doubling the previous one, are required")
    results = [smooth_annulus_spectrum([mode], parameters=parameters, radial_points=n, wall=wall,
                                       inner_wall=inner_wall, ring_edges=ring_edges,
                                       central_charge=central_charge)[0] for n in resolutions]
    rates = np.array([entry["growth_rate"] for entry in results])
    last = dict(results[-1])
    previous, latest = rates[-2]-rates[-3], rates[-1]-rates[-2]
    order = float(np.log2(previous/latest)) if previous*latest > 0 and abs(latest) < abs(previous) else None
    if order is not None and 1. <= order <= 4.:
        extrapolated = float(rates[-1]+latest/(2.**order-1.))
    else:
        order, extrapolated = None, float(rates[-1])
    correction = abs(extrapolated-rates[-1])
    relative = correction/max(abs(extrapolated), 1.e-14)
    last.update(growth_rate=extrapolated, raw_growth_rate=float(rates[-1]), refinements=results,
                richardson_order=order, refinement_absolute_change=correction,
                refinement_relative_change=relative,
                reference_converged=bool(order is not None and relative < 5.e-5),
                reference_type="smooth-radial-nystrom-richardson")
    return last


def truncated_annulus_spectrum(modes, *, parameters, radial_points=64, wall=1., inner_wall=0.):
    """Linear spectrum of the truncated super-Gaussian band of Zoni--Guclu (2019), eq. (35).

    rho0 = rho_bar exp(-|(r-s_bar)/s_d|^p) on [s_minus, s_plus], zero outside, so
    rho0' has a smooth part inside the band plus the jumps +-rho0(s_-+) at the cut
    radii. The perturbation is the smooth band density at Gauss nodes plus two
    surface charges at the cuts; with the Green function of
    :func:`annular_poisson_green` this is the Nystrom analogue of
    :func:`interface_spectrum` (recovered for p -> infinity) and of
    :func:`smooth_annulus_spectrum` (no cut). Each half band is split at
    0.1 s_d from the cut to resolve the steep layer of width ~ s_d/p there.
    """
    from scipy.linalg import eigvals

    inner, outer = float(parameters["s_minus"]), float(parameters["s_plus"])
    level = float(parameters.get("rho_bar", 1.))
    power = float(parameters["p"])
    centre = float(parameters.get("s_bar") or .5*(inner+outer))
    scale = float(parameters.get("s_d") or .5*(outer-inner))
    if not 0 <= inner_wall < inner < centre < outer < wall or power <= 0 or radial_points < 8:
        raise ValueError("require 0 <= inner_wall < s_minus < s_bar < s_plus < wall, p > 0, >= 8 points")

    def profile(r):
        r = np.asarray(r, dtype=float)
        inside = (r >= inner) & (r <= outer)
        u = np.minimum(np.abs((r-centre)/scale), 2.)  # outside values are discarded; avoid overflow
        return np.where(inside, level*np.exp(-u**power), 0.)

    def smooth_derivative(r):
        u = (np.asarray(r, dtype=float)-centre)/scale
        return -level*power/scale*np.sign(u)*np.abs(u)**(power-1)*np.exp(-np.abs(u)**power)

    layer = .1*scale
    edges = (inner, inner+layer, centre, outer-layer, outer)
    points, weights = np.polynomial.legendre.leggauss(int(radial_points))
    radii = np.concatenate([(a+b)/2+(b-a)*points/2 for a, b in zip(edges, edges[1:])])
    nodal = np.concatenate([weights*(b-a)/2 for a, b in zip(edges, edges[1:])])
    breaks = edges
    density = lambda r: float(profile(r))
    flux = _inner_wall_flux(density, inner_wall=float(inner_wall), wall=wall, breaks=breaks)
    omega = lambda r: (_enclosed_charge(r, density, inner_wall=float(inner_wall), breaks=breaks)-flux)/r**2
    omega_nodes = np.asarray([omega(r) for r in radii])
    omega_cuts = (omega(inner), omega(outer))
    jumps = (float(profile(inner)), -float(profile(outer)))  # rho0(r+) - rho0(r-) at the two cuts
    derivative = smooth_derivative(radii)/radii
    cuts = np.array([inner, outer])
    checked = interface_spectrum(modes, inner=inner, outer=outer, wall=wall, inner_wall=inner_wall, density=level)
    records = []
    for entry in checked:
        mode = entry["mode"]
        targets = np.concatenate([radii, cuts])
        sources = np.concatenate([radii*nodal, cuts])  # quadrature weights r_j w_j; surface charges at r = cut
        green = annular_poisson_green(mode, targets, np.concatenate([radii, cuts]),
                                      inner_wall=float(inner_wall), wall=wall)*sources[None, :]
        coupling = np.concatenate([derivative, np.array(jumps)/cuts])
        matrix = mode*coupling[:, None]*green
        matrix[np.diag_indices_from(matrix)] += mode*np.concatenate([omega_nodes, omega_cuts])
        eigenvalues = eigvals(matrix, overwrite_a=True, check_finite=False)
        fastest = eigenvalues[np.argmax(eigenvalues.imag)]
        growth = max(0., float(fastest.imag))
        records.append(dict(mode=mode, growth_rate=growth, omega_real=float(fastest.real),
                            unstable=bool(growth > 1.e-10), reference_type="truncated-profile-nystrom",
                            radial_points_per_segment=int(radial_points), radial_nodes=len(radii)+2))
    return records


_TRUNCATED_REFERENCE_CACHE: dict[str, dict] = {}


def converged_truncated_reference(mode, *, parameters, resolutions=(64, 128, 256), wall=1., inner_wall=0.):
    """Refined :func:`truncated_annulus_spectrum` with the last relative change (memoized)."""
    import json

    key = json.dumps(dict(mode=int(mode), parameters=parameters, resolutions=resolutions,
                          wall=float(wall), inner_wall=float(inner_wall)), sort_keys=True, default=str)
    if key in _TRUNCATED_REFERENCE_CACHE:
        return dict(_TRUNCATED_REFERENCE_CACHE[key])
    results = [truncated_annulus_spectrum([mode], parameters=parameters, radial_points=n, wall=wall,
                                          inner_wall=inner_wall)[0] for n in resolutions]
    rates = np.array([entry["growth_rate"] for entry in results])
    previous, latest = rates[-2]-rates[-3], rates[-1]-rates[-2]
    order = float(np.log2(previous/latest)) if previous*latest > 0 and abs(latest) < abs(previous) else None
    # Second-order Gauss-Nystrom (Green-function kink): Richardson-extrapolate when the order is clean.
    extrapolated = float(rates[-1]+latest/(2.**order-1.)) if order is not None and 1. <= order <= 4. else float(rates[-1])
    difference = abs(extrapolated-rates[-1])
    relative = difference/max(abs(extrapolated), 1.e-14)
    result = dict(results[-1], growth_rate=extrapolated, raw_growth_rate=float(rates[-1]), refinements=results,
                  richardson_order=order, refinement_absolute_change=difference,
                  refinement_relative_change=relative, reference_converged=bool(relative < 5.e-5))
    _TRUNCATED_REFERENCE_CACHE[key] = dict(result)
    return result


def point_vortex_ring_growth(n, central, *, radius=1., circulation=1.):
    """Largest growth rate of a ring of ``n`` point vortices around a central one.

    The ring has n vortices of circulation Gamma at radius R and the centre one of
    circulation ``central``*Gamma; the polygon rotates rigidly. The linearized
    (n+1)-vortex system (analytic Jacobian, rotating frame) gives the rate in
    1/time for the given Gamma and R (it scales as Gamma/R^2). Without a centre the
    ring is unstable for n >= 8 (Thomson); the centre stabilizes it inside
    :func:`point_vortex_ring_window`.
    """
    n, gamma, radius = int(n), float(circulation), float(radius)
    circulations = np.r_[np.full(n, gamma), float(central) * gamma]
    z = np.r_[radius * np.exp(2j * np.pi * np.arange(n) / n), 0.]
    omega = gamma * (n - 1) / (4 * np.pi * radius ** 2) + float(central) * gamma / (2 * np.pi * radius ** 2)
    m = n + 1
    # w_k = conj(dz_k/dt) = (1/2 pi i) sum_j Gamma_j/(z_k - z_j) is holomorphic in the positions.
    derivative = np.zeros((m, m), dtype=complex)
    for k in range(m):
        for j in range(m):
            if j != k:
                term = circulations[j] / (2j * np.pi) / (z[k] - z[j]) ** 2
                derivative[k, j] = term
                derivative[k, k] -= term
    jacobian = np.zeros((2 * m, 2 * m))
    jacobian[:m, :m], jacobian[:m, m:] = derivative.real, -derivative.imag
    jacobian[m:, :m], jacobian[m:, m:] = -derivative.imag, -derivative.real
    jacobian[np.arange(m), m + np.arange(m)] += omega
    jacobian[m + np.arange(m), np.arange(m)] -= omega
    return max(0., float(np.linalg.eigvals(jacobian).real.max()))


def point_vortex_ring_window(n):
    """Central circulation ratios (n^2 - 8n + 8)/16 < mu < (n - 1)^2/4 that stabilize a ring of n >= 7 vortices."""
    n = int(n)
    return (n * n - 8 * n + 8) / 16., (n - 1) ** 2 / 4.



class RingDispersion:
    """Dispersion relation omega(m) of a smooth disk ring, continued to complex m.

    The Nystrom operator of :func:`smooth_annulus_spectrum` (disk, ring breaks
    ``ring_edges``) with the Green function (r</r>)^m - (r r')^m)/(2m), which is
    analytic in m. ``__call__(m, guess)`` returns the eigenvalue closest to
    ``guess`` (the most unstable one without), so a branch is followed by
    continuation.
    """

    def __init__(self, parameters, *, radial_points=16, ring_edges=RING_EDGES, wall=1.):
        from scripts.guiding_center.cases.guiding_center_cases import diocotron_k

        import inspect

        accepted = inspect.signature(diocotron_k).parameters
        case = diocotron_k(**{key: value for key, value in parameters.items() if key in accepted})
        if case.equilibrium_radial_derivative is None:
            raise ValueError("the dispersion relation requires a smooth untruncated profile")
        center, width = case.parameters["s_bar"], case.parameters["s_d"]
        edges = {0., float(wall), case.parameters["s_minus"], case.parameters["s_plus"], center}
        edges |= {center + sign * float(w) * width for w in ring_edges for sign in (-1., 1.)}
        edges = sorted(e for e in edges if 0. <= e <= wall)
        points, weights = np.polynomial.legendre.leggauss(int(radial_points))
        radii = np.concatenate([(a + b) / 2 + (b - a) * points / 2 for a, b in zip(edges, edges[1:])])
        self.weights = np.concatenate([weights * (b - a) / 2 for a, b in zip(edges, edges[1:])]) * radii
        density = lambda s: case.equilibrium_density(s, 0)
        self.omega0 = np.array([_enclosed_charge(r, density, inner_wall=0., breaks=edges[1:-1]) / r ** 2
                                for r in radii])
        self.derivative = case.equilibrium_radial_derivative(radii) / radii
        r, q = radii[:, None], radii[None, :]
        self.log_ratio, self.log_product = np.log(np.minimum(r, q) / np.maximum(r, q)), np.log(r * q / wall ** 2)

    def __call__(self, m, guess=None):
        m = complex(m)
        green = (np.exp(m * self.log_ratio) - np.exp(m * self.log_product)) / (2 * m)
        matrix = m * self.derivative[:, None] * green * self.weights[None, :]
        matrix[np.diag_indices_from(matrix)] += m * self.omega0
        values = np.linalg.eigvals(matrix)
        return values[np.argmax(values.imag)] if guess is None else values[np.argmin(np.abs(values - guess))]


def pulled_front_speeds(dispersion, modes=range(1, 30), *, step=2.e-3, h=1.e-3):
    """Linear spreading speeds of a localized disturbance of a ring (lab frame, rad per time).

    Along a ray theta = v t the impulse response grows as exp(sigma(v) t) with
    sigma(v) = Im(omega(m*) - v m*) at the saddle omega'(m*) = v (complex m*).
    The edges sigma(v) = 0 are the speeds of fronts pulled by the linear
    instability (van Saarloos 2003). Continuation starts at the group velocity of
    the most unstable integer mode and steps v outwards. Returns the trailing and
    leading speeds, their saddles and the group velocity.
    """
    from scipy.optimize import brentq

    def derivatives(m, guess):
        value = dispersion(m, guess)
        plus, minus = dispersion(m + h, value), dispersion(m - h, value)
        return value, (plus - minus) / (2 * h), (plus - 2 * value + minus) / h ** 2

    def saddle(v, m, guess):
        for _ in range(40):
            value, first, second = derivatives(m, guess)
            update = (first - v) / second
            m, guess = m - update, value
            if abs(update) < 1.e-10:
                break
        value = dispersion(m, guess)
        return m, value, float((value - v * m).imag)

    rates = [(dispersion(m).imag, m) for m in modes]
    fastest = max(rates)[1]
    start_value, group, _ = derivatives(float(fastest), dispersion(fastest))
    result = dict(fastest_mode=int(fastest), group_velocity=float(group.real))
    for name, direction in (("leading", 1.), ("trailing", -1.)):
        v, m, value = float(group.real), complex(fastest), start_value
        previous = None
        for _ in range(int(1. / step)):
            m, value, sigma = saddle(v, m, value)
            if previous is not None and sigma < 0. < previous[0]:
                _, v0, m0, value0 = previous
                speed = brentq(lambda u: saddle(u, m0, value0)[2], v0, v, xtol=1.e-8)
                m_star, omega_star, _ = saddle(speed, m0, value0)
                result[name] = dict(speed=float(speed), saddle=complex(m_star), omega=complex(omega_star))
                break
            previous = (sigma, v, m, value)
            v += direction * step
        else:
            raise RuntimeError(f"no {name} edge found")
    return result


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
