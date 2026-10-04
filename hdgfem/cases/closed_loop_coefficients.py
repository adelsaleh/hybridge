"""Bundled analytic coefficients for corrugated-annulus ADR benchmarks.

The scalar/array formulas share geometric intermediates and work with NumPy,
CuPy's NumPy-ufunc dispatch, and Numba nopython compilation. Parameters are
``(hole_radius, epsilon, speed/normalization, reaction, variant)`` with variants
0=closed-loop trapping, 1=cross-stream transport, 2=constant-tensor orthogonal.
The domain must exclude the origin. No projection or numerical derivatives.
"""
import numpy as np

try:
    from numba.extending import register_jitable
except ImportError:
    def register_jitable(function):
        """Return ``function`` unchanged when Numba is unavailable (plain NumPy evaluation)."""
        return function


@register_jitable
def _geometry(x, y, hole_radius):
    """Return the annular radius and the corrugated stream coordinate at ``(x, y)``.

    ``(rho, rho_x, rho_y)`` is the radius normalized across the annulus width;
    ``chi`` holds the stream coordinate and its first and second derivatives.
    """
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    width = 1+0.35*np.cos(9*phi)-hole_radius
    wp, wpp = -3.15*np.sin(9*phi), -28.35*np.cos(9*phi)
    rho = (r-hole_radius)/width
    rx, ry, px, py = x/r, y/r, -y/r**2, x/r**2
    rxx, rxy, ryy = y*y/r**3, -x*y/r**3, x*x/r**3
    pxx, pxy, pyy = 2*x*y/r**4, (y*y-x*x)/r**4, -2*x*y/r**4
    dr, dp = 1/width, -rho*wp/width
    drp, dpp = -wp/width**2, rho*(2*(wp/width)**2-wpp/width)
    gx, gy = dr*rx+dp*px, dr*ry+dp*py
    hxx = 2*drp*rx*px+dpp*px*px+dr*rxx+dp*pxx
    hxy = drp*(rx*py+ry*px)+dpp*px*py+dr*rxy+dp*pxy
    hyy = 2*drp*ry*py+dpp*py*py+dr*ryy+dp*pyy
    a, q = 2*np.pi, 0.05*np.sin(5*phi)
    qp, qpp = 0.25*np.cos(5*phi), -1.25*np.sin(5*phi)
    sn, cs = np.sin(a*rho), np.cos(a*rho)
    fr, fp = 1+a*q*cs, qp*sn
    frr, frp, fpp = -a*a*q*sn, a*qp*cs, qpp*sn
    chi = (rho+q*sn, fr*gx+fp*px, fr*gy+fp*py,
           fr*hxx+fp*pxx+frr*gx*gx+2*frp*gx*px+fpp*px*px,
           fr*hxy+fp*pxy+frr*gx*gy+frp*(gx*py+gy*px)+fpp*px*py,
           fr*hyy+fp*pyy+frr*gy*gy+2*frp*gy*py+fpp*py*py)
    return (rho, gx, gy), chi


@register_jitable
def _velocity(x, y, radial, chi, parameters):
    """Return ``(beta_x, beta_y)`` for variant ``parameters[4]`` from :func:`_geometry` data.

    Variant 0 follows the closed level sets of ``chi``, 1 is a divergence-free
    cross-stream field from a stream function, and 2 is the constant-direction control.
    """
    _, _, scale, _, variant = parameters
    if variant == 2:
        c, s = np.cos(np.pi/7), np.sin(np.pi/7)
        g = scale*(1+0.8*np.sin(13*np.pi*(c*x+s*y)))/1.8
        return -s*g, c*g
    if variant == 0:
        factor = scale*np.sin(8*np.pi*chi[0])
        return factor*chi[2], -factor*chi[1]
    rho, rx, ry = radial
    envelope = 16*rho**2*(1-rho)**2
    derivative = 32*rho*(1-rho)*(1-2*rho)
    value, dx, dy = 0*x, 0*x, 0*x
    for amplitude, m, n, denominator in ((1.0, 13, 11, 13), (0.35, 17, 19, 19)):
        a, b, d = m*np.pi, n*np.pi, denominator*np.pi
        sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
        value = value+amplitude*sx*sy/d
        dx = dx+amplitude*a*cx*sy/d
        dy = dy+amplitude*b*sx*cy/d
    return scale*(derivative*ry*value+envelope*dy), -scale*(derivative*rx*value+envelope*dx)


@register_jitable
def closed_loop_velocity(x, y, parameters):
    """Return both velocity components together at volume or face points."""
    # No annular geometry is required for the constant-direction control case.
    if parameters[4] == 2:
        zero = 0*x
        return _velocity(x, y, (zero, zero, zero), (zero, zero, zero, zero, zero, zero), parameters)
    radial, chi = _geometry(x, y, parameters[0])
    return _velocity(x, y, radial, chi, parameters)


def closed_loop_volume(x, y, parameters):
    """Return ``Kxx, Kxy, Kyy, beta_x, beta_y, source`` in one evaluation."""
    hole_radius, eps, _, reaction, variant = parameters
    radial, chi = _geometry(x, y, hole_radius)
    _, gx, gy, hxx, hxy, hyy = chi
    if variant == 2:
        c, s = np.cos(np.pi/7), np.sin(np.pi/7)
        zero = 0*x
        kxx, kxy, kyy = zero+c*c+eps*s*s, zero+(1-eps)*c*s, zero+s*s+eps*c*c
        divx, divy = zero, zero
    else:
        norm = np.hypot(gx, gy)
        bx, by = gy/norm, -gx/norm
        nx, ny = (gx*hxx+gy*hxy)/norm, (gx*hxy+gy*hyy)/norm
        bxx, bxy = hxy/norm-bx*nx/norm, hyy/norm-bx*ny/norm
        byx, byy = -hxx/norm-by*nx/norm, -hxy/norm-by*ny/norm
        scale = 1-eps
        kxx, kxy, kyy = eps+scale*bx*bx, scale*bx*by, eps+scale*by*by
        divx = scale*(2*bx*bxx+bxy*by+bx*byy)
        divy = scale*(bxx*by+bx*byx+2*by*byy)
    vx, vy = _velocity(x, y, radial, chi, parameters)
    a = 2*np.pi
    u, du, ddu = np.sin(a*chi[0]), a*np.cos(a*chi[0]), -a*a*np.sin(a*chi[0])
    ux, uy = du*gx, du*gy
    uxx, uxy, uyy = du*hxx+ddu*gx*gx, du*hxy+ddu*gx*gy, du*hyy+ddu*gy*gy
    for amplitude, m, n in ((0.25, 6, 5), (0.25*0.35, 11, 9)):
        a, b = m*np.pi, n*np.pi
        sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
        v = amplitude*sx*sy
        u, ux, uy = u+v, ux+amplitude*a*cx*sy, uy+amplitude*b*sx*cy
        uxx, uxy, uyy = uxx-a*a*v, uxy+amplitude*a*b*cx*cy, uyy-b*b*v
    source = -kxx*uxx-2*kxy*uxy-kyy*uyy-divx*ux-divy*uy+vx*ux+vy*uy+reaction*u
    return kxx, kxy, kyy, vx, vy, source
