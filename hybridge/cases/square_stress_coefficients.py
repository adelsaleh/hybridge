"""Smooth square-domain counterpart of the closed-loop ADR stress family.

Domain: [-1, 1]^2. Scalar and broadcast-array formulas work with NumPy,
CuPy ufunc dispatch and Numba; importing this module does not compile anything.
Numeric parameters are (epsilon, speed / normalization, reaction, variant),
where variant is 0=trapping, 1=crossing, 2=constant-tensor orthogonal.

For trapping/crossing, K = epsilon I + (1-epsilon) t t^T/(|t|^2+delta^2),
where t=grad-perp(chi), chi=(1-x^2)(1-y^2), and delta=0.05.
Regularization is essential at the centre and corners: K is smooth and SPD
there, with eigenvalues epsilon and epsilon+(1-epsilon)|t|^2/(|t|^2+delta^2).
Thus anisotropy is bounded by 1/epsilon, not constant at stagnation points.
See docs/reference/square_stress_coefficients.md for the manufactured problem.
"""
import numpy as np

try:
    from numba.extending import register_jitable
except ImportError:
    def register_jitable(function):
        """Return ``function`` unchanged when Numba is unavailable (plain NumPy evaluation)."""
        return function


SQUARE_CORE = 0.05


@register_jitable
def square_coordinates(x, y):
    """Return chi and its analytic Cartesian gradient and Hessian."""
    a, b = 1-x*x, 1-y*y
    zero = 0*x+0*y
    return a*b, -2*x*b, -2*y*a, zero-2*b, 4*x*y, zero-2*a


@register_jitable
def square_exact_data(x, y):
    """Common exact scalar field, gradient and Hessian for all three cases."""
    chi, gx, gy, hxx, hxy, hyy = square_coordinates(x, y)
    a = 2*np.pi
    u, du, ddu = np.sin(a*chi), a*np.cos(a*chi), -a*a*np.sin(a*chi)
    ux, uy = du*gx, du*gy
    uxx, uxy, uyy = du*hxx+ddu*gx*gx, du*hxy+ddu*gx*gy, du*hyy+ddu*gy*gy
    for amplitude, m, n in ((0.25, 6, 5), (0.25*0.35, 11, 9)):
        a, b = m*np.pi, n*np.pi
        sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
        v = amplitude*sx*sy
        u, ux, uy = u+v, ux+amplitude*a*cx*sy, uy+amplitude*b*sx*cy
        uxx, uxy, uyy = uxx-a*a*v, uxy+amplitude*a*b*cx*cy, uyy-b*b*v
    return u, ux, uy, uxx, uxy, uyy


@register_jitable
def square_diffusion_data(x, y, parameters):
    """Return Kxx, Kxy, Kyy, (div K)x and (div K)y, without polar coordinates."""
    eps, _, _, variant = parameters
    if variant == 2:
        c, s = np.cos(np.pi/7), np.sin(np.pi/7)
        zero = 0*x+0*y
        return zero+c*c+eps*s*s, zero+(1-eps)*c*s, zero+s*s+eps*c*c, zero, zero
    _, gx, gy, hxx, hxy, hyy = square_coordinates(x, y)
    tx, ty = gy, -gx
    txx, txy, tyx, tyy = hxy, hyy, -hxx, -hxy
    norm2 = tx*tx+ty*ty+SQUARE_CORE*SQUARE_CORE
    nx, ny = 2*(tx*txx+ty*tyx), 2*(tx*txy+ty*tyy)
    scale = (1-eps)/norm2
    divx = scale*(2*tx*txx+txy*ty+tx*tyy-(tx*tx*nx+tx*ty*ny)/norm2)
    divy = scale*(txx*ty+tx*tyx+2*ty*tyy-(tx*ty*nx+ty*ty*ny)/norm2)
    return eps+scale*tx*tx, scale*tx*ty, eps+scale*ty*ty, divx, divy


@register_jitable
def square_velocity(x, y, parameters):
    """Divergence-free velocity, including smooth centre/corner limits."""
    _, scale, _, variant = parameters
    if variant == 2:
        c, s = np.cos(np.pi/7), np.sin(np.pi/7)
        g = scale*(1+0.8*np.sin(13*np.pi*(c*x+s*y)))/1.8
        return -s*g, c*g
    chi, gx, gy, _, _, _ = square_coordinates(x, y)
    if variant == 0:
        factor = scale*np.sin(8*np.pi*chi)
        return factor*gy, -factor*gx
    # psi_cross=chi^2*q. Both psi and its gradient vanish on the square walls.
    value, dx, dy = 0*x+0*y, 0*x+0*y, 0*x+0*y
    for amplitude, m, n, denominator in ((1.0, 13, 11, 13), (0.35, 17, 19, 19)):
        a, b, d = m*np.pi, n*np.pi, denominator*np.pi
        sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
        value = value+amplitude*sx*sy/d
        dx = dx+amplitude*a*cx*sy/d
        dy = dy+amplitude*b*sx*cy/d
    return scale*(2*chi*gy*value+chi*chi*dy), -scale*(2*chi*gx*value+chi*chi*dx)


def square_volume(x, y, parameters):
    """Bundled sampler contract: Kxx, Kxy, Kyy, beta_x, beta_y, source."""
    u, ux, uy, uxx, uxy, uyy = square_exact_data(x, y)
    kxx, kxy, kyy, divx, divy = square_diffusion_data(x, y, parameters)
    vx, vy = square_velocity(x, y, parameters)
    source = (-kxx*uxx-2*kxy*uxy-kyy*uyy-divx*ux-divy*uy
              +vx*ux+vy*uy+parameters[2]*u)
    return kxx, kxy, kyy, vx, vy, source
