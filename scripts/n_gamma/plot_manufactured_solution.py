import matplotlib
matplotlib.use("qt5agg")   # put this BEFORE importing pyplot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# Domain / time interval
# ============================================================

Nx = Ny = 300
x = np.linspace(-1.0, 1.0, Nx)
y = np.linspace(-1.0, 1.0, Ny)
X, Y = np.meshgrid(x, y, indexing="xy")

T0, Tf = 0.0, 2.0 * np.pi
nframes = 200
times = np.linspace(T0, Tf, nframes)


# ============================================================
# Manufactured solution
# ============================================================

def density(t, x, y):
    return (
        2.0
        + 0.2 * np.sin(np.pi * x - t)
              * np.cos(np.pi * y + 2.0 * t)
        + 0.1 * np.cos(2.0 * np.pi * x + t)
              * np.sin(np.pi * y - t)
    )


def velocity(t, x, y):
    return (
        0.2
        + 0.4 * np.cos(np.pi * x + 2.0 * t)
              * np.sin(np.pi * y - t)
        + 0.1 * np.sin(2.0 * np.pi * x - t)
              * np.cos(np.pi * y + 3.0 * t)
    )


def momentum(t, x, y):
    n = density(t, x, y)
    u = velocity(t, x, y)
    return n * u


# ============================================================
# Compute fixed plotting ranges over the whole animation
# ============================================================

# Sampling 100 times is more than enough to determine useful
# global plotting bounds.
sample_times = np.linspace(T0, Tf, 100)

n_min = np.inf
n_max = -np.inf
g_min = np.inf
g_max = -np.inf

for t in sample_times:
    N = density(t, X, Y)
    G = momentum(t, X, Y)

    n_min = min(n_min, N.min())
    n_max = max(n_max, N.max())
    g_min = min(g_min, G.min())
    g_max = max(g_max, G.max())

print(f"n range     : [{n_min:.6f}, {n_max:.6f}]")
print(f"Gamma range : [{g_min:.6f}, {g_max:.6f}]")

# Symmetric Gamma scale around zero is useful because Gamma changes sign.
g_abs = max(abs(g_min), abs(g_max))
gamma_norm = TwoSlopeNorm(vmin=-g_abs, vcenter=0.0, vmax=g_abs)


# ============================================================
# Figure
# ============================================================

fig, (ax_n, ax_g) = plt.subplots(
    1, 2,
    figsize=(12, 5.2),
    constrained_layout=True
)

N0 = density(times[0], X, Y)
G0 = momentum(times[0], X, Y)

im_n = ax_n.imshow(
    N0,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    interpolation="bilinear",
    aspect="equal",
    cmap="viridis",
    vmin=n_min,
    vmax=n_max,
)

im_g = ax_g.imshow(
    G0,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    interpolation="bilinear",
    aspect="equal",
    cmap="RdBu_r",
    norm=gamma_norm,
)

cb_n = fig.colorbar(im_n, ax=ax_n, shrink=0.88)
cb_g = fig.colorbar(im_g, ax=ax_g, shrink=0.88)

cb_n.set_label(r"$n$")
cb_g.set_label(r"$\Gamma$")

ax_n.set_title(r"Density $n(t,x,y)$")
ax_g.set_title(r"Momentum $\Gamma(t,x,y)=n\,u$")

for ax in (ax_n, ax_g):
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

time_text = fig.suptitle(r"$t=0.000$", fontsize=14)


# ============================================================
# Animation
# ============================================================

def update(frame):
    t = times[frame]

    N = density(t, X, Y)
    G = momentum(t, X, Y)

    im_n.set_data(N)
    im_g.set_data(G)

    time_text.set_text(
        rf"$t={t:.3f}\qquad (t/2\pi={t/(2*np.pi):.2f})$"
    )

    return im_n, im_g, time_text


ani = FuncAnimation(
    fig,
    update,
    frames=nframes,
    interval=40,
    blit=False,
    repeat=True,
)

plt.show()


# ============================================================
# Optional: save animation
# ============================================================

# MP4 (requires ffmpeg):
# ani.save(
#     "manufactured_n_gamma.mp4",
#     writer="ffmpeg",
#     fps=30,
#     dpi=160,
# )

# GIF (requires pillow):
# ani.save(
#     "manufactured_n_gamma.gif",
#     writer="pillow",
#     fps=25,
# )