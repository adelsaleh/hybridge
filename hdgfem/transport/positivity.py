r"""Mass-conserving KKT projection of a DG density onto nonnegativity at element points.

For a density :math:`\rho^*` (for example the unconstrained result of one
implicit transport step) :class:`DensityPositivityProjector` returns a density
that is nonnegative at a fixed point set :math:`S_K` of every element and has
the same total mass. Element by element, with the element mass matrix
:math:`M_K = J_K M_{ref}`:

* a nonnegative element mean is kept exactly, and the element becomes the
  :math:`M_K`-closest polynomial that is nonnegative on :math:`S_K` (the
  minimal-norm counterpart of Zhang-Shu scaling, always feasible);
* a negative element mean is replaced by the :math:`M_K`-closest polynomial
  that is nonnegative on :math:`S_K`, which adds a small mass;
* the added mass is returned by one global factor :math:`1-\eta`, which keeps
  every sign at :math:`S_K`.

Each element problem is a least-distance problem
:math:`\min\|w\|_2\ \mathrm{s.t.}\ W w \ge h`, solved by the Lawson-Hanson
active-set method on its NNLS form. Every active-set step is a Newton step on
the KKT system restricted to the active constraints; the step-length safeguard
and linearly independent active sets keep it finite on the degenerate active
sets of near-zero elements, where more points than modes are active. Elements
whose mean is negligible take their constant mean, the only feasible
polynomial; Zhang-Shu scaling is the always-feasible fallback, and a final
exact scaling toward the mean removes rounding-level violations.

Positivity is enforced at points, not everywhere: point constraints keep the
O(h^(p+1)) projection error, whereas Bernstein-coefficient constraints reduce
it to about O(h^2.5) near zeros of the density. See
``docs/development/plans/positivity_kkt_bdf2.md``.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from hdgfem.core.space import DGField, DGSpace
from hdgfem.runtime.optional import njit, prange
from hdgfem.runtime.precision import REAL_DTYPE

POINT_SETS = ("quadrature+lattice", "quadrature+lattice+dense")

# Status codes per projected element.
_PROJECTED, _CONSTANT, _FALLBACK, _FREE, _FREE_FALLBACK = range(5)


def _lattice(order: int) -> np.ndarray:
    """Equispaced order-``order`` lattice of the reference triangle ``(-1, 1)``."""
    return np.array([(-1 + 2*i/order, -1 + 2*j/order)
                     for i in range(order + 1) for j in range(order + 1 - i)], dtype=float)


def positivity_points(space: DGSpace, points: str = "quadrature+lattice") -> np.ndarray:
    """Return the reference evaluation table ``V`` (points x modes) of a point set.

    ``quadrature+lattice``: the volume quadrature points of ``space`` plus the
    equispaced p-lattice (vertices, edge and interior nodes).
    ``quadrature+lattice+dense`` also adds the (2p+2)-lattice sampled by
    :class:`hdgfem.diagnostics.guiding_center.ScalarPositivityDiagnostics`.
    """
    if points not in POINT_SETS:
        raise ValueError(f"points must be one of {POINT_SETS}")
    order = max(int(space.order), 1)
    tables = [space.quad_data.bas_of_quads.T, space.reference.basis_at(_lattice(order))]
    if points.endswith("+dense"):
        tables.append(space.reference.basis_at(_lattice(2 * order + 2)))
    return np.ascontiguousarray(np.vstack(tables), dtype=float)


class _ReferenceTables:
    """Shared affine-reference tables for the element least-distance problems."""

    def __init__(self, space: DGSpace, points: str):
        """Build the point table, the mass factor and both reduced constraint systems."""
        q = space.quad_data
        weights = np.asarray(q.Krf_w, dtype=float)
        basis = np.asarray(q.bas_of_quads, dtype=float)              # (modes, quadrature)
        mass = (basis * weights) @ basis.T                             # reference mass matrix
        self.V = positivity_points(space, points)                      # (points, modes)
        self.modes = self.V.shape[1]
        self.integrals = basis @ weights                               # ∫_ref φ_i
        self.area = float(weights.sum())
        self.mass = np.ascontiguousarray(mass)
        self.constant = np.linalg.solve(mass, self.integrals)          # coefficients of f ≡ 1
        lower = np.linalg.cholesky(mass)
        inverse_transpose = np.linalg.inv(lower).T                     # x - y = L^{-T} z
        # Mean-preserving variables: z orthogonal to a = L^{-1} ∫φ, through an
        # orthonormal complement Q (modes x modes-1).
        a = np.linalg.solve(lower, self.integrals)
        q_full, _ = np.linalg.qr(np.column_stack([a, np.eye(self.modes)]))
        complement = q_full[:, 1:self.modes]
        self.R_mean = np.ascontiguousarray(inverse_transpose @ complement)
        self.R_free = np.ascontiguousarray(inverse_transpose)
        self.W_mean = np.ascontiguousarray(self.V @ self.R_mean)
        self.W_free = np.ascontiguousarray(self.V @ self.R_free)
        self.G_mean = np.ascontiguousarray(self.W_mean @ self.W_mean.T)
        self.G_free = np.ascontiguousarray(self.W_free @ self.W_free.T)


@njit(cache=True)
def _factor_passive(G, h, index, active, L):
    """Cholesky factor of E_P^T E_P = G_PP + h_P h_P^T for the first ``active`` indices."""
    for i in range(active):
        ii = index[i]
        for j in range(i + 1):
            jj = index[j]
            value = G[ii, jj] + h[ii] * h[jj]
            for k in range(j):
                value -= L[i, k] * L[j, k]
            if i == j:
                if not value > 1e-300:
                    return False
                L[i, i] = np.sqrt(value)
            else:
                L[i, j] = value / L[j, j]
    return True


@njit(cache=True)
def _append_passive(G, h, index, active, L):
    """Add the row of index ``index[active]`` to the factor by one forward solve (O(p^2))."""
    tt = index[active]
    total = 0.0
    for i in range(active):
        ii = index[i]
        value = G[ii, tt] + h[ii] * h[tt]
        for k in range(i):
            value -= L[i, k] * L[active, k]
        value /= L[i, i]
        L[active, i] = value
        total += value * value
    diagonal = G[tt, tt] + h[tt] * h[tt] - total
    if not diagonal > 1e-300:
        return False
    L[active, active] = np.sqrt(diagonal)
    return True


@njit(cache=True)
def _solve_passive(L, h, index, active, out):
    """Solve (L L^T) s = h_P with the passive factor."""
    for i in range(active):
        value = h[index[i]]
        for k in range(i):
            value -= L[i, k] * out[k]
        out[i] = value / L[i, i]
    for i in range(active - 1, -1, -1):
        value = out[i]
        for k in range(i + 1, active):
            value -= L[k, i] * out[k]
        out[i] = value / L[i, i]


@njit(cache=True)
def _least_distance(G, W, h, result, u, passive, index, L, s_p):
    """Lawson-Hanson NNLS solution of min ||w|| s.t. W w >= h (``h`` scaled to max |h| = 1).

    The NNLS matrix is E = [W^T; h^T] with target e_last, so E_P^T E_P =
    G_PP + h_P h_P^T and E_P^T f = h_P. The passive Cholesky factor grows by one
    row per added constraint and is rebuilt only after constraints are dropped.
    Returns (status, iterations): 0 on success, 1 if infeasible, 2 on a
    numerical breakdown or iteration limit.
    """
    m = h.shape[0]
    capacity = index.shape[0]
    diagonal = 0.0
    for i in range(m):
        u[i] = 0.0
        passive[i] = False
        diagonal = max(diagonal, G[i, i])
    tolerance = 1e-12 * (1.0 + diagonal)
    active = 0
    iterations = 0
    converged = False
    for outer in range(3 * m + 10):
        hu = 0.0
        for p in range(active):
            hu += h[index[p]] * u[index[p]]
        best, chosen = tolerance, -1
        for i in range(m):
            if passive[i]:
                continue
            gradient = h[i] * (1.0 - hu)
            for p in range(active):
                gradient -= G[i, index[p]] * u[index[p]]
            if gradient > best:
                best, chosen = gradient, i
        if chosen < 0:
            converged = True
            break
        if active == capacity:
            return 2, iterations
        passive[chosen] = True
        index[active] = chosen
        if not _append_passive(G, h, index, active, L):
            return 2, iterations
        active += 1
        solved = False
        for inner in range(3 * m + 10):
            iterations += 1
            _solve_passive(L, h, index, active, s_p)
            positive = True
            for a in range(active):
                if s_p[a] <= 0.0:
                    positive = False
                    break
            if positive:
                for a in range(active):
                    u[index[a]] = s_p[a]
                solved = True
                break
            alpha = 2.0
            for a in range(active):
                if s_p[a] <= 0.0:
                    current = u[index[a]]
                    alpha = min(alpha, current / (current - s_p[a]))
            kept = 0
            for a in range(active):
                ia = index[a]
                value = u[ia] + alpha * (s_p[a] - u[ia])
                if value <= 1e-300:
                    u[ia] = 0.0
                    passive[ia] = False
                else:
                    u[ia] = value
                    index[kept] = ia
                    kept += 1
            active = kept
            if active == 0:
                solved = True
                break
            if not _factor_passive(G, h, index, active, L):
                return 2, iterations
        if not solved:
            return 2, iterations
    if not converged:
        return 2, iterations
    hu = 0.0
    for p in range(active):
        hu += h[index[p]] * u[index[p]]
    denominator = 1.0 - hu
    if not denominator > 1e-14:
        return 1, iterations
    for c in range(W.shape[1]):
        value = 0.0
        for p in range(active):
            value += W[index[p], c] * u[index[p]]
        result[c] = value / denominator
    return 0, iterations


@njit(cache=True)
def _clamp_toward_mean(x, V, constant, mean):
    """Scale ``x`` toward its constant mean until it is nonnegative at every point."""
    lowest = 0.0
    for i in range(V.shape[0]):
        value = 0.0
        for j in range(V.shape[1]):
            value += V[i, j] * x[j]
        lowest = min(lowest, value)
    if lowest < 0.0:
        theta = mean / (mean - lowest) if mean > 0.0 else 0.0
        for j in range(x.shape[0]):
            x[j] = mean * constant[j] + theta * (x[j] - mean * constant[j])


@njit(cache=True, parallel=True)
def _project_elements_host(coefficients, flagged, V, integrals, area, constant, mass,
                           G_mean, W_mean, R_mean, G_free, W_free, R_free,
                           mean_tolerance, status, iterations):
    """Project the flagged elements of ``coefficients`` in place (Numba, parallel)."""
    modes = coefficients.shape[1]
    m = V.shape[0]
    for e in prange(flagged.shape[0]):
        K = flagged[e]
        y = coefficients[K].copy()
        b = V @ y
        scale = np.max(np.abs(b))
        mean = 0.0
        for j in range(modes):
            mean += integrals[j] * y[j]
        mean /= area
        x = y.copy()
        u = np.empty(m)
        passive = np.empty(m, dtype=np.bool_)
        index = np.empty(modes + 1, dtype=np.int64)
        L = np.empty((modes + 1, modes + 1))
        s_p = np.empty(modes + 1)
        h = -b / scale
        if mean >= 0.0:
            if mean <= mean_tolerance * scale:
                for j in range(modes):
                    x[j] = mean * constant[j]
                status[e] = _CONSTANT
            else:
                w = np.empty(R_mean.shape[1])
                code, count = _least_distance(G_mean, W_mean, h, w, u, passive, index, L, s_p)
                iterations[e] = count
                lowest = np.min(b)
                theta = mean / (mean - lowest)
                difference = mean * constant - y
                fallback = (1.0 - theta) ** 2 * (difference @ (mass @ difference))
                if code == 0 and scale * scale * (w @ w) <= fallback * (1.0 + 1e-12) + 1e-300:
                    x = y + scale * (R_mean @ w)
                    status[e] = _PROJECTED
                else:
                    x = mean * constant + theta * (y - mean * constant)
                    status[e] = _FALLBACK
                _clamp_toward_mean(x, V, constant, mean)
        else:
            z = np.empty(R_free.shape[1])
            code, count = _least_distance(G_free, W_free, h, z, u, passive, index, L, s_p)
            iterations[e] = count
            if code == 0:
                x = y + scale * (R_free @ z)
                status[e] = _FREE
            else:
                x[:] = 0.0
                status[e] = _FREE_FALLBACK
            new_mean = 0.0
            for j in range(modes):
                new_mean += integrals[j] * x[j]
            new_mean /= area
            if new_mean <= 0.0:
                x[:] = 0.0
                status[e] = _FREE_FALLBACK
            else:
                _clamp_toward_mean(x, V, constant, new_mean)
        coefficients[K] = x


_DEVICE_SOURCE = r"""
// One warp per element. Lanes share the point loops; the passive Cholesky factor,
// the dual iterate and the element data live in per-warp shared memory.
#define MODES __MODES__
#define POINTS __POINTS__
#define CAPACITY (MODES + 1)
#define WARPS __WARPS__
#define FULL 0xffffffffu

struct WarpState {
    double L[CAPACITY * CAPACITY];
    double y[MODES], x[MODES], w[MODES];
    double b[POINTS], h[POINTS], u[POINTS];
    double s_p[CAPACITY], r[CAPACITY];
    int index[CAPACITY];
    unsigned char passive[POINTS];
};

__device__ __forceinline__ double warp_sum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(FULL, value, offset);
    return value;
}
__device__ __forceinline__ double warp_max(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmax(value, __shfl_xor_sync(FULL, value, offset));
    return value;
}
__device__ __forceinline__ double warp_min(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmin(value, __shfl_xor_sync(FULL, value, offset));
    return value;
}

// Forward then backward column sweeps: solve (L L^T) out = rhs for the first n rows.
__device__ void solve_factor(WarpState& s, int n, const double* rhs, double* out, int lane) {
    for (int i = lane; i < n; i += 32) s.r[i] = rhs[i];
    __syncwarp();
    for (int k = 0; k < n; ++k) {
        const double value = s.r[k] / s.L[k * CAPACITY + k];
        __syncwarp();
        for (int i = k + 1 + lane; i < n; i += 32) s.r[i] -= s.L[i * CAPACITY + k] * value;
        if (lane == 0) s.r[k] = value;
        __syncwarp();
    }
    for (int k = n - 1; k >= 0; --k) {
        const double value = s.r[k] / s.L[k * CAPACITY + k];
        __syncwarp();
        for (int i = lane; i < k; i += 32) s.r[i] -= s.L[k * CAPACITY + i] * value;
        if (lane == 0) s.r[k] = value;
        __syncwarp();
    }
    for (int i = lane; i < n; i += 32) out[i] = s.r[i];
    __syncwarp();
}

// Factor G_PP + h_P h_P^T for the first n passive indices (column-oriented, lanes over rows).
__device__ bool factor_passive(WarpState& s, const double* __restrict__ G, int n, int lane) {
    for (int j = 0; j < n; ++j) {
        const int jj = s.index[j];
        for (int i = j + lane; i < n; i += 32) {
            const int ii = s.index[i];
            double value = G[ii * POINTS + jj] + s.h[ii] * s.h[jj];
            for (int k = 0; k < j; ++k) value -= s.L[i * CAPACITY + k] * s.L[j * CAPACITY + k];
            s.L[i * CAPACITY + j] = value;
        }
        __syncwarp();
        const double diagonal = s.L[j * CAPACITY + j];
        if (!(diagonal > 1e-300)) return false;
        const double root = sqrt(diagonal);
        __syncwarp();
        for (int i = j + 1 + lane; i < n; i += 32) s.L[i * CAPACITY + j] /= root;
        if (lane == 0) s.L[j * CAPACITY + j] = root;
        __syncwarp();
    }
    return true;
}

// Append the row of s.index[n] to the factor of the first n indices: one forward sweep.
__device__ bool append_passive(WarpState& s, const double* __restrict__ G, int n, int lane) {
    const int tt = s.index[n];
    for (int i = lane; i < n; i += 32) {
        const int ii = s.index[i];
        s.r[i] = G[ii * POINTS + tt] + s.h[ii] * s.h[tt];
    }
    __syncwarp();
    for (int k = 0; k < n; ++k) {
        const double value = s.r[k] / s.L[k * CAPACITY + k];
        __syncwarp();
        for (int i = k + 1 + lane; i < n; i += 32) s.r[i] -= s.L[i * CAPACITY + k] * value;
        if (lane == 0) s.r[k] = value;
        __syncwarp();
    }
    double partial = 0.0;
    for (int i = lane; i < n; i += 32) { s.L[n * CAPACITY + i] = s.r[i]; partial += s.r[i] * s.r[i]; }
    const double diagonal = G[tt * POINTS + tt] + s.h[tt] * s.h[tt] - warp_sum(partial);
    if (!(diagonal > 1e-300)) return false;
    if (lane == 0) s.L[n * CAPACITY + n] = sqrt(diagonal);
    __syncwarp();
    return true;
}

// Lawson-Hanson NNLS for min ||w|| s.t. W w >= h (h scaled to max |h| = 1), as on the host.
__device__ int least_distance(WarpState& s, const double* __restrict__ G, const double* __restrict__ W,
                              int columns, int lane, int* iterations) {
    double diagonal = 0.0;
    for (int i = lane; i < POINTS; i += 32) {
        s.u[i] = 0.0; s.passive[i] = 0; diagonal = fmax(diagonal, G[i * POINTS + i]);
    }
    const double tolerance = 1e-12 * (1.0 + warp_max(diagonal));
    __syncwarp();
    int active = 0, count = 0;
    bool converged = false;
    for (int outer = 0; outer < 3 * POINTS + 10; ++outer) {
        double partial = 0.0;
        for (int p = lane; p < active; p += 32) partial += s.h[s.index[p]] * s.u[s.index[p]];
        const double hu = warp_sum(partial);
        double best = tolerance; int chosen = -1;
        for (int i = lane; i < POINTS; i += 32) {
            if (s.passive[i]) continue;
            double gradient = s.h[i] * (1.0 - hu);
            for (int p = 0; p < active; ++p) gradient -= G[i * POINTS + s.index[p]] * s.u[s.index[p]];
            if (gradient > best || (gradient == best && chosen >= 0 && i < chosen)) { best = gradient; chosen = i; }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other = __shfl_xor_sync(FULL, best, offset);
            const int other_index = __shfl_xor_sync(FULL, chosen, offset);
            if (other_index >= 0 && (other > best || (other == best && (chosen < 0 || other_index < chosen)))) {
                best = other; chosen = other_index;
            }
        }
        if (chosen < 0) { converged = true; break; }
        if (active == CAPACITY) { *iterations = count; return 2; }
        if (lane == 0) { s.passive[chosen] = 1; s.index[active] = chosen; }
        __syncwarp();
        if (!append_passive(s, G, active, lane)) { *iterations = count; return 2; }
        ++active;
        bool solved = false;
        for (int inner = 0; inner < 3 * POINTS + 10; ++inner) {
            ++count;
            double* rhs = s.s_p;  // rhs = h_P; solve_factor copies it before overwriting
            for (int a = lane; a < active; a += 32) rhs[a] = s.h[s.index[a]];
            __syncwarp();
            solve_factor(s, active, rhs, s.s_p, lane);
            const bool lane_positive = (lane >= active) || (s.s_p[lane] > 0.0);
            if (__all_sync(FULL, lane_positive)) {
                for (int a = lane; a < active; a += 32) s.u[s.index[a]] = s.s_p[a];
                __syncwarp();
                solved = true; break;
            }
            double alpha = 2.0;
            if (lane < active && s.s_p[lane] <= 0.0) {
                const double current = s.u[s.index[lane]];
                alpha = current / (current - s.s_p[lane]);
            }
            alpha = warp_min(alpha);
            bool keep = false; int ia = -1; double value = 0.0;
            if (lane < active) {
                ia = s.index[lane];
                value = s.u[ia] + alpha * (s.s_p[lane] - s.u[ia]);
                keep = value > 1e-300;
            }
            const unsigned kept_mask = __ballot_sync(FULL, keep);
            __syncwarp();
            if (lane < active) {
                if (keep) {
                    s.u[ia] = value;
                    s.index[__popc(kept_mask & ((1u << lane) - 1u))] = ia;
                } else {
                    s.u[ia] = 0.0; s.passive[ia] = 0;
                }
            }
            __syncwarp();
            active = __popc(kept_mask);
            if (active == 0) { solved = true; break; }
            if (!factor_passive(s, G, active, lane)) { *iterations = count; return 2; }
        }
        if (!solved) { *iterations = count; return 2; }
    }
    *iterations = count;
    if (!converged) return 2;
    double partial = 0.0;
    for (int p = lane; p < active; p += 32) partial += s.h[s.index[p]] * s.u[s.index[p]];
    const double denominator = 1.0 - warp_sum(partial);
    if (!(denominator > 1e-14)) return 1;
    for (int c = lane; c < columns; c += 32) {
        double value = 0.0;
        for (int p = 0; p < active; ++p) value += W[s.index[p] * columns + c] * s.u[s.index[p]];
        s.w[c] = value / denominator;
    }
    __syncwarp();
    return 0;
}

__device__ double point_minimum(WarpState& s, const double* __restrict__ V, int lane) {
    double lowest = 0.0;
    for (int i = lane; i < POINTS; i += 32) {
        double value = 0.0;
        for (int j = 0; j < MODES; ++j) value += V[i * MODES + j] * s.x[j];
        lowest = fmin(lowest, value);
    }
    return warp_min(lowest);
}

__device__ void clamp_toward_mean(WarpState& s, const double* __restrict__ V, const double* __restrict__ constant,
                                  double mean, int lane) {
    const double lowest = point_minimum(s, V, lane);
    if (lowest < 0.0) {
        const double theta = mean > 0.0 ? mean / (mean - lowest) : 0.0;
        __syncwarp();
        for (int j = lane; j < MODES; j += 32) s.x[j] = mean * constant[j] + theta * (s.x[j] - mean * constant[j]);
    }
    __syncwarp();
}

extern "C" __global__ void project_density_elements(
        double* coefficients, const long long* flagged, const int count,
        const double* __restrict__ V, const double* __restrict__ integrals, const double area,
        const double* __restrict__ constant, const double* __restrict__ mass,
        const double* __restrict__ G_mean, const double* __restrict__ W_mean, const double* __restrict__ R_mean,
        const double* __restrict__ G_free, const double* __restrict__ W_free, const double* __restrict__ R_free,
        const double mean_tolerance, int* status, int* iterations) {
    extern __shared__ unsigned char workspace[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x * WARPS + warp;
    if (e >= count) return;
    WarpState& s = reinterpret_cast<WarpState*>(workspace)[warp];
    double* row = coefficients + flagged[e] * MODES;
    double partial = 0.0;
    for (int j = lane; j < MODES; j += 32) { s.y[j] = row[j]; s.x[j] = row[j]; partial += integrals[j] * row[j]; }
    const double mean = warp_sum(partial) / area;
    __syncwarp();
    double scale = 0.0, lowest = 0.0;
    for (int i = lane; i < POINTS; i += 32) {
        double value = 0.0;
        for (int j = 0; j < MODES; ++j) value += V[i * MODES + j] * s.y[j];
        s.b[i] = value; scale = fmax(scale, fabs(value)); lowest = fmin(lowest, value);
    }
    scale = warp_max(scale); lowest = warp_min(lowest);
    for (int i = lane; i < POINTS; i += 32) s.h[i] = -s.b[i] / scale;
    __syncwarp();
    int code_status = 0, count_iterations = 0;
    if (mean >= 0.0) {
        if (mean <= mean_tolerance * scale) {
            for (int j = lane; j < MODES; j += 32) s.x[j] = mean * constant[j];
            code_status = 1;
            __syncwarp();
        } else {
            const int code = least_distance(s, G_mean, W_mean, MODES - 1, lane, &count_iterations);
            const double theta = mean / (mean - lowest);
            double fallback = 0.0, norm = 0.0;
            for (int i = lane; i < MODES; i += 32) {
                double value = 0.0;
                for (int j = 0; j < MODES; ++j) value += mass[i * MODES + j] * (mean * constant[j] - s.y[j]);
                fallback += (mean * constant[i] - s.y[i]) * value;
            }
            fallback = warp_sum(fallback) * (1.0 - theta) * (1.0 - theta);
            if (code == 0) {
                for (int c = lane; c < MODES - 1; c += 32) norm += s.w[c] * s.w[c];
                norm = warp_sum(norm);
            }
            if (code == 0 && scale * scale * norm <= fallback * (1.0 + 1e-12) + 1e-300) {
                for (int i = lane; i < MODES; i += 32) {
                    double value = 0.0;
                    for (int c = 0; c < MODES - 1; ++c) value += R_mean[i * (MODES - 1) + c] * s.w[c];
                    s.x[i] = s.y[i] + scale * value;
                }
                code_status = 0;
            } else {
                for (int i = lane; i < MODES; i += 32) s.x[i] = mean * constant[i] + theta * (s.y[i] - mean * constant[i]);
                code_status = 2;
            }
            __syncwarp();
            clamp_toward_mean(s, V, constant, mean, lane);
        }
    } else {
        const int code = least_distance(s, G_free, W_free, MODES, lane, &count_iterations);
        if (code == 0) {
            for (int i = lane; i < MODES; i += 32) {
                double value = 0.0;
                for (int c = 0; c < MODES; ++c) value += R_free[i * MODES + c] * s.w[c];
                s.x[i] = s.y[i] + scale * value;
            }
            code_status = 3;
        } else {
            for (int i = lane; i < MODES; i += 32) s.x[i] = 0.0;
            code_status = 4;
        }
        __syncwarp();
        double partial_mean = 0.0;
        for (int j = lane; j < MODES; j += 32) partial_mean += integrals[j] * s.x[j];
        const double new_mean = warp_sum(partial_mean) / area;
        if (new_mean <= 0.0) {
            for (int i = lane; i < MODES; i += 32) s.x[i] = 0.0;
            code_status = 4;
            __syncwarp();
        } else {
            clamp_toward_mean(s, V, constant, new_mean, lane);
        }
    }
    for (int j = lane; j < MODES; j += 32) row[j] = s.x[j];
    if (lane == 0) { status[e] = code_status; iterations[e] = count_iterations; }
}
"""

_DEVICE_WARPS_PER_BLOCK = 4


class DensityPositivityProjector:
    """Project DG densities onto nonnegativity at element points, conserving mass.

    ``points`` selects the constrained point set (:func:`positivity_points`);
    ``backend`` is ``"host"`` (NumPy/Numba) or ``"device"`` (CuPy and a raw CUDA
    kernel, coefficients stay resident). The mesh must be affine. ``project``
    returns the projected field and a small report of scalar diagnostics.
    """

    def __init__(self, space: DGSpace, *, points: str = "quadrature+lattice", backend: str = "host",
                 mean_tolerance: float = 1e-8, detection_tolerance: float = 1e-13,
                 negligible_tolerance: float = 1e-12):
        """Cache the reference tables, element Jacobians and, on the device, the kernel.

        An element is projected when its minimum at the points is below
        ``-detection_tolerance`` times its largest absolute point value, so
        rounding-level values of an already projected density are left alone.
        A flagged element whose largest point value is below ``negligible_tolerance``
        times the field's largest is scaled toward its mean (Zhang-Shu) instead of
        solving its KKT problem: at that size optimality is irrelevant, and these
        near-zero far-field elements are the ones that cost the most iterations.
        """
        if backend not in {"host", "device"}:
            raise ValueError("backend must be 'host' or 'device'")
        for value in (mean_tolerance, detection_tolerance, negligible_tolerance):
            if not np.isfinite(value) or value < 0:
                raise ValueError("tolerances must be finite and nonnegative")
        self.space, self.points, self.backend = space, points, backend
        self.mean_tolerance = float(mean_tolerance)
        self.detection_tolerance = float(detection_tolerance)
        self.negligible_tolerance = float(negligible_tolerance)
        self.tables = _ReferenceTables(space, points)
        self.jacobians = np.ascontiguousarray(space.mesh.aff_jacs, dtype=float)
        self._device = None
        if backend == "device":
            self._device = self._prepare_device()

    def _prepare_device(self):
        """Upload the shared tables and compile the per-element kernel."""
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy_device

        cp = require_cupy_device()
        cspace = as_cupy_space(self.space)
        t = self.tables
        with cp.cuda.Device(cspace.device_id):
            arrays = {name: cp.asarray(getattr(t, name), dtype=cp.float64) for name in (
                "V", "integrals", "constant", "mass", "G_mean", "W_mean", "R_mean",
                "G_free", "W_free", "R_free")}
            arrays["jacobians"] = cp.asarray(self.jacobians)
            source = (_DEVICE_SOURCE.replace("__MODES__", str(t.modes))
                      .replace("__POINTS__", str(t.V.shape[0]))
                      .replace("__WARPS__", str(_DEVICE_WARPS_PER_BLOCK)))
            kernel = cp.RawKernel(source, "project_density_elements", options=("--std=c++11",))
            capacity, modes, points = t.modes + 1, t.modes, t.V.shape[0]
            state = 8 * (capacity * capacity + 3 * modes + 3 * points + 2 * capacity) + 4 * capacity + points
            state = (state + 15) // 16 * 16
            shared = state * _DEVICE_WARPS_PER_BLOCK
            kernel.max_dynamic_shared_size_bytes = shared
        return dict(cp=cp, cspace=cspace, arrays=arrays, kernel=kernel, shared=shared)

    def project(self, field: DGField, *, name: str | None = None):
        """Return ``(projected_field, report)`` for one density field."""
        if not isinstance(field, DGField) or field.space is not self.space:
            raise ValueError("the projector requires a DGField on its own DGSpace")
        start = perf_counter()
        if self.backend == "device":
            projected, report = self._project_device(field, name)
        else:
            projected, report = self._project_host(field, name)
        report["positivity_projection_time"] = perf_counter() - start
        return projected, report

    def _scale_negligible(self, xp, y, x, values, flagged):
        """Zhang-Shu-scale flagged elements of negligible size; return the rest and the count.

        Scaling toward a positive mean by mean / (mean - min) makes every point value
        nonnegative and keeps the element mean; a nonpositive mean becomes zero and
        its mass is returned with the others.
        """
        if not self.negligible_tolerance or not flagged.size:
            return flagged, 0
        local = values[flagged]
        tiny = xp.abs(local).max(axis=1) < self.negligible_tolerance * xp.abs(values).max()
        chosen = flagged[tiny]
        if chosen.size:
            arrays = self._device["arrays"] if xp is not np else None
            integrals = self.tables.integrals if xp is np else arrays["integrals"]
            constant = self.tables.constant if xp is np else arrays["constant"]
            mean = (y[chosen] @ integrals) / self.tables.area
            low = local[tiny].min(axis=1)
            theta = xp.where(mean > 0, mean / xp.maximum(mean - low, 1e-300), 0.0)
            centre = xp.maximum(mean, 0.0)[:, None] * constant[None, :]
            x[chosen] = centre + theta[:, None] * (y[chosen] - centre)
        return flagged[~tiny], int(chosen.size)

    def _summary(self, xp, y, x, b_min, changed, status, iterations, excess, total, negligible=0):
        """Pack the scalar report from the before/after coefficients."""
        t = self.tables
        jac = self.jacobians if xp is np else self._device["arrays"]["jacobians"]
        mass = t.mass if xp is np else self._device["arrays"]["mass"]
        V = t.V if xp is np else self._device["arrays"]["V"]
        d = x - y
        scalar = lambda value: xp.asarray(value, dtype=xp.float64).reshape(())
        norm = lambda c: xp.sum(jac[:, None] * ((c @ mass) * c))
        after = (x[changed] @ V.T).min() if changed.size else 0.0
        counts = [scalar(xp.count_nonzero(status == code)) for code in range(5)]
        packed = xp.stack([
            scalar(changed.size), *counts, scalar(negligible),
            scalar(iterations.max() if iterations.size else 0),
            scalar(b_min), scalar(after),
            xp.sqrt(norm(d) / xp.maximum(norm(y), 1e-300)), scalar(excess / total)])
        values = packed.get() if xp is not np else packed
        # Status codes: 0 projected, 1 constant mean, 2 Zhang-Shu fallback,
        # 3 negative mean projected, 4 negative mean set to zero; negligible elements
        # were scaled toward their mean before the element solves.
        keys = ("positivity_flagged", "positivity_projected", "positivity_constant",
                "positivity_fallback", "positivity_negative_mean_projected",
                "positivity_negative_mean_zeroed", "positivity_negligible", "positivity_max_iterations",
                "positivity_min_before", "positivity_min_after",
                "positivity_correction_relative", "positivity_mass_returned")
        report = {key: float(value) for key, value in zip(keys, values)}
        for key in keys[:8]:
            report[key] = int(report[key])
        return report

    def _mass_return(self, xp, x, y, jac):
        """Apply the global factor that returns the mass added by negative-mean elements."""
        integrals = self.tables.integrals if xp is np else self._device["arrays"]["integrals"]
        before = jac @ (y @ integrals)
        after = jac @ (x @ integrals)
        excess = after - before
        if float(after) <= 0.0:
            raise ValueError("cannot project a density with nonpositive total mass")
        x *= 1.0 - excess / after
        return excess, before

    def _project_host(self, field, name):
        """NumPy detection, Numba element solves, NumPy mass return."""
        t = self.tables
        y = np.ascontiguousarray(field.coeffs, dtype=float)
        values = y @ t.V.T
        minima = values.min(axis=1)
        changed = np.flatnonzero(minima < -self.detection_tolerance * np.abs(values).max(axis=1)).astype(np.int64)
        x = y.copy()
        flagged, negligible = self._scale_negligible(np, y, x, values, changed)
        status = np.full(flagged.size, -1, dtype=np.int64)
        iterations = np.zeros(flagged.size, dtype=np.int64)
        if flagged.size:
            _project_elements_host(x, flagged, t.V, t.integrals, t.area, t.constant, t.mass,
                                   t.G_mean, t.W_mean, t.R_mean, t.G_free, t.W_free, t.R_free,
                                   self.mean_tolerance, status, iterations)
        excess, total = self._mass_return(np, x, y, self.jacobians)
        report = self._summary(np, y, x, minima.min() if minima.size else 0.0, changed,
                               status, iterations, excess, total, negligible)
        projected = self.space.field(x.astype(REAL_DTYPE, copy=False), name=name or field.name)
        return projected, report

    def _project_device(self, field, name):
        """Device detection, kernel element solves and mass return; coefficients stay resident."""
        from hdgfem.core.device import as_cupy_coefficients, field_from_cupy_coefficients

        device = self._device
        cp, cspace, arrays, kernel = device["cp"], device["cspace"], device["arrays"], device["kernel"]
        with cp.cuda.Device(cspace.device_id):
            y = cp.ascontiguousarray(as_cupy_coefficients(field, cspace), dtype=cp.float64)
            values = y @ arrays["V"].T
            minima = values.min(axis=1)
            changed = cp.flatnonzero(
                minima < -self.detection_tolerance * cp.abs(values).max(axis=1)).astype(cp.int64)
            x = y.copy()
            flagged, negligible = self._scale_negligible(cp, y, x, values, changed)
            del values
            count = int(flagged.size)
            status = cp.full(count, -1, dtype=cp.int32)
            iterations = cp.zeros(count, dtype=cp.int32)
            if count:
                warps = _DEVICE_WARPS_PER_BLOCK
                kernel(((count + warps - 1) // warps,), (32 * warps,), (
                    x, flagged, np.int32(count), arrays["V"], arrays["integrals"],
                    np.float64(self.tables.area), arrays["constant"], arrays["mass"],
                    arrays["G_mean"], arrays["W_mean"], arrays["R_mean"],
                    arrays["G_free"], arrays["W_free"], arrays["R_free"],
                    np.float64(self.mean_tolerance), status, iterations), shared_mem=device["shared"])
            excess, total = self._mass_return(cp, x, y, arrays["jacobians"])
            report = self._summary(cp, y, x, minima.min() if minima.size else cp.asarray(0.0),
                                   changed, status, iterations, excess, total, negligible)
            projected = field_from_cupy_coefficients(
                self.space, x.astype(REAL_DTYPE, copy=False), device=cspace.device_id,
                name=name or field.name)
        return projected, report


__all__ = ["DensityPositivityProjector", "POINT_SETS", "positivity_points"]
