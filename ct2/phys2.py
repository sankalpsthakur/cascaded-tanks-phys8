"""Round-2 grey-box Cascaded Tanks simulator and fitter (JAX, float64).

A superset of ct/phys.py (which stays frozen and untouched). For every structure that ct/phys.py can express (pure physics, no
residual MLP, no dead zone, no priors) this module computes the same trajectory (checked by ct2/selftest.py). New in round 2:

  tp   first-order pump/inflow lag (samples). The lag state xp starts at u_hist[1] (steady state, adds no fitted number).
       Exact exponential update per RK4 sub-step; the sub-step uses the exact mean of the lagged input.
  to   first-order delay (samples) of the routed upper-tank overflow volume before it reaches the lower tank (buffer state xb, starts at 0).
  w3   level-dependent lower-outlet coefficient: q3 = k3 exp(w3 (x2 - 5) / 5) sqrt(x2).

Fitting options (Fitter): plain MSE or pseudo-Huber loss (huber_delta, in volts), single shooting or multiple shooting
(ms_K equal segments, continuity penalty ms_lam in sample-equivalents), warm starts, per-start diagnostics.
Time unit = 1 sample; levels in sensor volts.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

XM1 = 10.0
EPS = 1e-4
Y_CLIP = 10.0

# name: (transform, default). The default is the value used when the parameter is not free, and the centre of random starts.
PARAMS: Dict[str, Tuple[str, float]] = {
    "k1": ("pos", 0.20), "r2": ("pos", 1.00), "k3": ("pos", 0.20), "k4": ("pos", 0.18), "beta": ("pos", 0.50),
    "a": ("real", 0.0), "b4": ("real", 0.0), "p1": ("pos", 0.5), "p3": ("pos", 0.5), "c2": ("real", 0.0),
    "cA": ("real", 0.0), "c4": ("real", 0.0), "cA1": ("real", 0.0), "h0": ("real", 0.0), "qov": ("pos", 1.0),
    "d1": ("real", -3.0), "d2": ("real", -6.0),
    "tp": ("pos", 1.0), "to": ("pos", 1.0), "w3": ("real", 0.0),
}
INIT_SCALE = {"a": 0.8, "b4": 0.5, "c2": 0.01, "cA": 0.05, "c4": 0.02, "cA1": 0.05, "h0": 1.5}
# unconstrained value of a new parameter that switches its term (practically) off; used for warm starts of nested candidates
NEUTRAL_Z = {"tp": float(np.log(0.02)), "to": float(np.log(0.02)), "w3": 0.0, "qov": 0.0, "c4": 0.0, "cA1": 0.0, "h0": 0.0,
             "d1": -12.0, "d2": -12.0}


@dataclasses.dataclass(frozen=True)
class Structure:
    """`free` is normalised to the order of PARAMS: unpack() assigns the entries of z in that order, so init, warm starts and
    unpacking agree for any order the caller writes."""
    free: Tuple[str, ...] = ("k1", "r2", "k3", "k4", "beta", "a", "b4", "cA")
    clamp_x2: bool = True
    nsub: int = 4
    name: str = ""

    def __post_init__(self):
        order = list(PARAMS)
        object.__setattr__(self, "free", tuple(sorted(self.free, key=order.index)))

    def key(self):
        return (self.free, self.clamp_x2, self.nsub)


def _to_con(name, z):
    return jnp.exp(z) if PARAMS[name][0] == "pos" else z


def unpack(struct: Structure, zfree, z_x0):
    th = {}
    zi = 0
    for n, (t, d) in PARAMS.items():
        if n in struct.free:
            th[n] = _to_con(n, zfree[zi]); zi += 1
        else:
            th[n] = jnp.asarray(d)
    return th, jnp.exp(z_x0)


def build_sim(struct: Structure):
    nsub = struct.nsub
    h = 1.0 / nsub
    F = struct.free
    use_h0, use_qov, use_w3 = "h0" in F, "qov" in F, "w3" in F
    use_tp, use_to = "tp" in F, "to" in F
    use_fir = "d1" in F or "d2" in F
    free_p1, free_p3 = "p1" in F, "p3" in F
    use_cA1, use_c4, use_c2 = "cA1" in F, "c4" in F, "c2" in F

    def rhs(x1, x2, u, th):
        x1 = jnp.maximum(x1, 0.0)
        x2 = jnp.maximum(x2, 0.0)
        xe1 = (jnp.maximum(x1 - th["h0"], 0.0) if use_h0 else x1) + EPS
        q1 = th["k1"] * (xe1 ** th["p1"] if free_p1 else jnp.sqrt(xe1))
        uu = u - th["b4"]
        inflow = th["k4"] * (uu + th["c4"] * uu ** 2) if use_c4 else th["k4"] * uu
        dx1 = inflow - q1
        if use_cA1:
            dx1 = dx1 / jnp.maximum(1.0 + th["cA1"] * (x1 - 5.0), 0.2)
        xe2 = x2 + EPS
        q3 = th["k3"] * (xe2 ** th["p3"] if free_p3 else jnp.sqrt(xe2))
        if use_w3:
            q3 = q3 * jnp.exp(th["w3"] * (x2 - 5.0) / 5.0)
        cap = jnp.maximum(1.0 + th["cA"] * (x2 - 5.0), 0.2)
        dx2 = (th["r2"] * q1 - q3) / cap
        return dx1, dx2

    def step(x, ut, th):
        """x = [x1, x2, xp, xb]; ut = (FIR-filtered) input of this sample; returns the state one sample later."""
        x1, x2, xp, xb = x[0], x[1], x[2], x[3]
        for _ in range(nsub):
            if use_tp:
                e = jnp.exp(-h / th["tp"])
                ueff = ut + (xp - ut) * (th["tp"] / h) * (1.0 - e)
                xp = ut + (xp - ut) * e
            else:
                ueff = ut
            a1, a2 = rhs(x1, x2, ueff, th)
            b1, b2 = rhs(x1 + 0.5 * h * a1, x2 + 0.5 * h * a2, ueff, th)
            c1, c2 = rhs(x1 + 0.5 * h * b1, x2 + 0.5 * h * b2, ueff, th)
            d1, d2 = rhs(x1 + h * c1, x2 + h * c2, ueff, th)
            x1 = x1 + (h / 6.0) * (a1 + 2 * b1 + 2 * c1 + d1)
            x2 = x2 + (h / 6.0) * (a2 + 2 * b2 + 2 * c2 + d2)
            sur = jnp.maximum(x1 - XM1, 0.0)
            x1 = x1 - sur
            if use_qov:
                routed = th["beta"] * h * (sur / h + 1e-6) ** th["qov"] - th["beta"] * h * 1e-6 ** th["qov"]
            else:
                routed = th["beta"] * sur
            if use_to:
                xb = xb + routed
                deliv = xb * (-jnp.expm1(-h / th["to"]))
                xb = xb - deliv
                x2 = x2 + deliv
            else:
                x2 = x2 + routed
            if struct.clamp_x2:
                x2 = jnp.minimum(x2, Y_CLIP - th["a"])
            x1 = jnp.maximum(x1, 0.0)
            x2 = jnp.maximum(x2, 0.0)
        return jnp.stack([x1, x2, xp, xb])

    def out(x, th):
        x2 = x[1]
        return jnp.minimum(th["a"] + x2 + (th["c2"] * x2 ** 2 if use_c2 else 0.0), Y_CLIP)

    def simulate(th, x0, u, u_hist):
        """u: (T,), u_hist: (2,) = [u_{-2}, u_{-1}]. Returns (y (T,), states (T,2) before each step, final state (2,))."""
        if use_fir:
            w = jax.nn.softmax(jnp.stack([jnp.asarray(0.0), th["d1"], th["d2"]]))
            u_pad = jnp.concatenate([u_hist, u])
            u_eff = w[0] * u_pad[2:] + w[1] * u_pad[1:-1] + w[2] * u_pad[:-2]
        else:
            u_eff = u
        xfull = jnp.stack([x0[0], x0[1], u_hist[1], jnp.asarray(0.0)])

        def body(x, ut):
            return step(x, ut, th), (out(x, th), x[:2])

        xl, (ys, xs) = jax.lax.scan(body, xfull, u_eff)
        return ys, xs, xl[:2]

    return simulate


def pseudo_huber(e, delta):
    """2 delta^2 (sqrt(1 + (e/delta)^2) - 1): equals e^2 for |e| << delta, grows like 2 delta |e| in the tails."""
    return 2.0 * delta ** 2 * (jnp.sqrt(1.0 + (e / delta) ** 2) - 1.0)


class Fitter:
    """Single-record simulation-error fitting with L-BFGS-B (JAX gradients), multi-start, optional warm starts.

    loss = 'mse' | 'huber'; ms_K > 0 switches to multiple shooting with K equal segments, each with its own free initial state, and the
    penalty ms_lam * sum(gap^2) / T on the state mismatch at the segment boundaries (gap in volts; ms_lam in sample-equivalents).
    The parameter vector z = [free parameters (log for positive ones), log x0 (2 numbers; K x 2 numbers for multiple shooting)].
    """

    def __init__(self, struct: Structure, loss: str = "mse", huber_delta: float = 0.2, ms_K: int = 0, ms_lam: float = 30.0):
        self.s = struct
        self.loss_kind, self.delta, self.K, self.lam = loss, huber_delta, ms_K, ms_lam
        self.sim = build_sim(struct)
        self.nfree = len(struct.free)
        nfree, sim, K = self.nfree, self.sim, ms_K

        def rho(e):
            return pseudo_huber(e, huber_delta) if loss == "huber" else e ** 2

        def loss_ss(z, u, y, w, u_hist):
            th, x0 = unpack(struct, z[:nfree], z[nfree:nfree + 2])
            yh, _, _ = sim(th, x0, u, u_hist)
            return jnp.sum(w * rho(yh - y)) / jnp.sum(w)

        def loss_ms(z, u, y, w, u_hist):
            T = u.shape[0]
            L = T // K
            th, _ = unpack(struct, z[:nfree], z[nfree:nfree + 2])
            zs = z[nfree:nfree + 2 * K].reshape(K, 2)
            x0s = jnp.exp(zs)
            us = u.reshape(K, L)
            uh = jnp.concatenate([u_hist[None, :], jnp.stack([u[k * L - 2:k * L] for k in range(1, K)])], axis=0)
            ys, _, xend = jax.vmap(lambda x0, uk, hk: sim(th, x0, uk, hk))(x0s, us, uh)
            data = jnp.sum(w * rho(ys.reshape(-1) - y)) / jnp.sum(w)
            gaps = xend[:-1] - x0s[1:]
            return data + ms_lam * jnp.sum(gaps ** 2) / T

        loss_fn = loss_ms if K > 0 else loss_ss
        self.vg = jax.jit(jax.value_and_grad(loss_fn))
        self.loss = jax.jit(loss_fn)

        def sim_fn(z, u, u_hist):
            th, x0 = unpack(struct, z[:nfree], z[nfree:nfree + 2])
            return sim(th, x0, u, u_hist)[0]

        self.simj = jax.jit(sim_fn)

        def states_fn(z, u, u_hist):
            th, x0 = unpack(struct, z[:nfree], z[nfree:nfree + 2])
            return sim(th, x0, u, u_hist)[1]

        self.statesj = jax.jit(states_fn)

        def win_loss(zx, zf, u, y, u_hist):
            th, x0 = unpack(struct, zf, zx)
            yh, _, _ = sim(th, x0, u, u_hist)
            return jnp.mean((yh - y) ** 2)

        self.win_vg = jax.jit(jax.value_and_grad(win_loss))
        self.win_loss = jax.jit(win_loss)

    # -- parameter helpers -------------------------------------------------
    def init_z(self, rng: np.random.Generator, y0: float, jitter=0.35):
        """Random start. The draw order is that of ct/phys.py, so the base parameters of a nested structure get the same draws."""
        zf = []
        for n in self.s.free:
            t, d = PARAMS[n]
            if n in ("d1", "d2"):
                zf.append(d + rng.normal(0, 1.0))
            elif t == "pos":
                zf.append(np.log(d) + rng.normal(0, jitter))
            else:
                zf.append(d + rng.normal(0, INIT_SCALE.get(n, 0.3)))
        x2 = max(y0 + rng.normal(0, 0.3), 0.5)
        x1 = np.clip(rng.uniform(2, 9), 0.5, 9.9)
        return np.array(zf + [np.log(x1), np.log(x2)], float)

    def extend_ms(self, z, u, u_hist):
        """Single-shooting vector (nfree + 2) -> multiple-shooting vector: segment start states from the simulation of z."""
        if self.K == 0:
            return np.asarray(z, float)
        z = np.asarray(z, float)
        T = len(u)
        L = T // self.K
        xs = np.asarray(self.statesj(z[: self.nfree + 2], np.asarray(u, float), np.asarray(u_hist, float)))
        starts = np.log(np.maximum(xs[[k * L for k in range(self.K)]], 0.1))
        starts[0] = z[self.nfree:self.nfree + 2]
        return np.concatenate([z[: self.nfree], starts.reshape(-1)])

    def fit(self, u, y, w=None, n_starts=4, seed=0, maxiter=400, u_hist=None, warm=(), verbose=False):
        """Returns (z, loss, diag). Starts: the given warm vectors (single-shooting layout, nfree + 2 numbers) first, then n_starts random
        starts drawn like ct/phys.py (default_rng(seed), one init_z per start). The best start by final loss wins."""
        u = np.asarray(u, float); y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        uh = np.array([u[0], u[0]]) if u_hist is None else np.asarray(u_hist, float)
        rng = np.random.default_rng(seed)
        starts = [("warm", np.asarray(z0, float)) for z0 in warm]
        for k in range(n_starts):
            starts.append(("rand", self.init_z(rng, y[0])))
        best, diag = None, []
        for kind, z0 in starts:
            t0 = time.time()
            try:
                z0 = self.extend_ms(z0, u, uh)
                res = minimize(lambda z: tuple(np.asarray(v, float) for v in self.vg(z, u, y, w, uh)),
                               z0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=maxiter * 2))
            except Exception as e:  # numerical failure in a start
                diag.append(dict(kind=kind, ok=False, err=repr(e)[:200]))
                continue
            ok = bool(np.isfinite(res.fun))
            diag.append(dict(kind=kind, ok=ok, loss=float(res.fun), nit=int(res.nit), nfev=int(res.nfev), secs=round(time.time() - t0, 2)))
            if verbose:
                print(f"  start {kind}: loss {res.fun:.5f} nit {res.nit}", flush=True)
            if ok and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("all starts failed: " + repr(diag))
        return best.x, float(best.fun), diag

    def fit_x0(self, z, u_win, y_win, u_hist, n_starts=12, seed=0):
        """Window fit of the initial state only, parameters frozen at z[:nfree]. Grid + polish. Uses only the given window."""
        zf = jnp.asarray(z[: self.nfree])
        best = None
        y0 = float(y_win[0])
        a_ = self._a(z)
        cands = []
        for x1 in (1.5, 3.0, 5.0, 7.0, 9.0):
            for dx in (-0.3, 0.3):
                cands.append(np.array([np.log(x1), np.log(max(y0 + dx - a_, 0.3))]))
        for z0 in cands[:n_starts]:
            res = minimize(lambda zx: tuple(np.asarray(v, float) for v in self.win_vg(zx, zf, np.asarray(u_win, float), np.asarray(y_win, float), np.asarray(u_hist, float))),
                           z0, jac=True, method="L-BFGS-B", options=dict(maxiter=100))
            if best is None or res.fun < best.fun:
                best = res
        return best.x, float(best.fun)

    def _a(self, z):
        return float(z[self.s.free.index("a")]) if "a" in self.s.free else PARAMS["a"][1]

    def simulate_with_init(self, z, zx, u, u_hist):
        zz = np.concatenate([np.asarray(z[: self.nfree]), np.asarray(zx)])
        return np.asarray(self.simj(zz, np.asarray(u, float), np.asarray(u_hist, float)))

    def theta(self, z):
        th, x0 = unpack(self.s, jnp.asarray(z[: self.nfree]), jnp.asarray(z[self.nfree:self.nfree + 2]))
        return {k: float(v) for k, v in th.items()}, np.asarray(x0)


def block_rmse(yh, y, n_init):
    e = np.asarray(yh)[n_init:] - np.asarray(y)[n_init:]
    return float(np.sqrt(np.mean(e ** 2)))
