"""Grey-box Cascaded Tanks simulator in JAX (float64): two-tank Torricelli ODE with overflow routing, sensor clip,
RK4 sub-stepping, zero-order-hold input. Pure functions; parameters are a flat vector with named transforms.

Time unit = 1 sample (Ts = 4 s is absorbed into the rate constants). Level units = sensor volts.
Free structure knobs live in `Structure`; everything not free is fixed at its default.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

XM1 = 10.0  # upper-tank overflow level fixed to 10 (scale of x1 is otherwise unidentifiable)
EPS = 1e-4
Y_CLIP = 10.0

# name: (transform, default)
PARAMS: Dict[str, Tuple[str, float]] = {
    "k1": ("pos", 0.20),   # upper tank outflow gain
    "r2": ("pos", 1.00),   # inflow-to-lower / outflow-of-upper ratio (area ratio)
    "k3": ("pos", 0.20),   # lower tank outflow gain
    "k4": ("pos", 0.18),   # pump gain
    "beta": ("pos", 0.50),  # fraction of surplus (overflowing) volume routed to the lower tank
    "a": ("real", 0.0),    # sensor offset: y = a + x2
    "b4": ("real", 0.0),   # pump offset (dead voltage)
    "p1": ("pos", 0.5),    # exponent of upper outflow
    "p3": ("pos", 0.5),    # exponent of lower outflow
    "c2": ("real", 0.0),   # sensor curvature: y = a + x2 + c2 x2^2
    "cA": ("real", 0.0),   # lower-tank capacity slope: dx2 /= (1 + cA (x2 - 5))  (bottle-shaped tank)
    "c4": ("real", 0.0),   # pump curvature: inflow = k4 (uu + c4 uu^2), uu = u - b4
    "cA1": ("real", 0.0),  # upper-tank capacity slope: dx1 /= (1 + cA1 (x1 - 5))
    "h0": ("real", 0.0),   # upper-tank outlet height: q1 = k1 sqrt(max(x1 - h0, 0))
    "qov": ("pos", 1.0),   # overflow routing exponent: routed volume = beta * h * (surplus rate)^qov (jet clears the lower tank only at high flow)
    "d1": ("real", -3.0),  # input FIR tap logits (softmax over taps 0,1,2) -- tap-1 logit
    "d2": ("real", -6.0),  # tap-2 logit (tap 0 logit is 0)
}


@dataclasses.dataclass(frozen=True)
class Structure:
    free: Tuple[str, ...] = ("k1", "k3", "k4", "beta")
    tie_r2: bool = True          # r2 == 1 if not free
    clamp_x2: bool = True        # lower tank full at x2 = 10 - a (state clamp) else only the sensor clips at 10
    nsub: int = 4
    deadzone: bool = False       # pump delivers nothing below b4 (softplus dead zone) instead of a signed linear offset
    res_hidden: int = 0          # 0 = pure physics; >0 adds a tanh-MLP residual on the state derivative
    res_l2: float = 1e-3         # L2 weight on the residual MLP parameters
    prior: Tuple[Tuple[str, float, float], ...] = ()   # (name, mean, scale) Gaussian priors on constrained params (e.g. exponents ~ 0.5)
    prior_w: float = 0.0         # weight of the prior penalty in MSE units
    name: str = ""

    def key(self):
        return (self.free, self.tie_r2, self.clamp_x2, self.nsub, self.deadzone, self.res_hidden, self.prior, self.prior_w)


def _to_con(name, z):
    t, _ = PARAMS[name]
    return jnp.exp(z) if t == "pos" else z


def _to_unc(name, v):
    t, _ = PARAMS[name]
    return float(np.log(v)) if t == "pos" else float(v)


def n_res(struct: Structure) -> int:
    h = struct.res_hidden
    return 0 if h == 0 else 3 * h + h + h * 2 + 2


def unpack(struct: Structure, zfree, z_x0):
    th = {}
    zi = 0
    for n, (t, d) in PARAMS.items():
        if n in struct.free:
            th[n] = _to_con(n, zfree[zi]); zi += 1
        else:
            th[n] = jnp.asarray(d)
    h = struct.res_hidden
    if h > 0:
        th["W1"] = zfree[zi: zi + 3 * h].reshape(3, h); zi += 3 * h
        th["b1"] = zfree[zi: zi + h]; zi += h
        th["W2"] = zfree[zi: zi + 2 * h].reshape(h, 2); zi += 2 * h
        th["b2"] = zfree[zi: zi + 2]; zi += 2
    x0 = jnp.exp(z_x0)  # positive states
    return th, x0


def build_sim(struct: Structure):
    nsub = struct.nsub
    h = 1.0 / nsub

    def rhs(x, u, th):
        x1 = jnp.maximum(x[0], 0.0)
        x2 = jnp.maximum(x[1], 0.0)
        q1 = th["k1"] * (jnp.maximum(x1 - th["h0"], 0.0) + EPS) ** th["p1"] if "h0" in struct.free else th["k1"] * (x1 + EPS) ** th["p1"]
        uu = u - th["b4"]
        if struct.deadzone:
            uu = 0.08 * jax.nn.softplus(uu / 0.08)
        cap1 = jnp.maximum(1.0 + th["cA1"] * (x1 - 5.0), 0.2)
        dx1 = (th["k4"] * (uu + th["c4"] * uu ** 2) - q1) / cap1
        cap = jnp.maximum(1.0 + th["cA"] * (x2 - 5.0), 0.2)
        dx2 = (th["r2"] * q1 - th["k3"] * (x2 + EPS) ** th["p3"]) / cap
        d = jnp.stack([dx1, dx2])
        if struct.res_hidden > 0:
            z = jnp.stack([x1 / 10.0, x2 / 10.0, (u - 3.0) / 3.0])
            d = d + 0.1 * (jnp.tanh(z @ th["W1"] + th["b1"]) @ th["W2"] + th["b2"])
        return d

    def clamp(x, th):
        sur = jnp.maximum(x[0] - XM1, 0.0)
        x1 = x[0] - sur
        if "qov" in struct.free:
            routed = th["beta"] * h * (sur / h + 1e-6) ** th["qov"] - th["beta"] * h * 1e-6 ** th["qov"]
        else:
            routed = th["beta"] * sur
        x2 = x[1] + routed
        if struct.clamp_x2:
            x2 = jnp.minimum(x2, Y_CLIP - th["a"])
        return jnp.stack([jnp.maximum(x1, 0.0), jnp.maximum(x2, 0.0)])

    def step(x, u, th):
        for _ in range(nsub):
            k1_ = rhs(x, u, th)
            k2_ = rhs(x + 0.5 * h * k1_, u, th)
            k3_ = rhs(x + 0.5 * h * k2_, u, th)
            k4_ = rhs(x + h * k3_, u, th)
            x = x + (h / 6.0) * (k1_ + 2 * k2_ + 2 * k3_ + k4_)
            x = clamp(x, th)
        return x

    def out(x, th):
        return jnp.minimum(th["a"] + x[1] + th["c2"] * x[1] ** 2, Y_CLIP)

    def simulate(th, x0, u, u_hist):
        """u: (T,), u_hist: (2,) = [u_{-2}, u_{-1}] samples preceding u[0] (used by the input FIR). Returns y (T,), states (T,2)."""
        use_fir = "d1" in struct.free or "d2" in struct.free
        if use_fir:
            logits = jnp.stack([jnp.asarray(0.0), th["d1"], th["d2"]])
            w = jax.nn.softmax(logits)
            u_pad = jnp.concatenate([u_hist, u])
            u_eff = w[0] * u_pad[2:] + w[1] * u_pad[1:-1] + w[2] * u_pad[:-2]
        else:
            u_eff = u

        def body(x, ut):
            y = out(x, th)
            return step(x, ut, th), (y, x)

        _, (ys, xs) = jax.lax.scan(body, x0, u_eff)
        return ys, xs

    return simulate


class Fitter:
    """Single-record simulation-error fitting with L-BFGS (jax gradients) and multi-start."""

    def __init__(self, struct: Structure):
        self.s = struct
        self.sim = build_sim(struct)
        self.nfree = len(struct.free) + n_res(struct)
        nfree, sim = self.nfree, self.sim
        nres = n_res(struct)
        l2 = struct.res_l2

        def loss_fn(z, u, y, w, u_hist):
            th, x0 = unpack(struct, z[:nfree], z[nfree:])
            yh, _ = sim(th, x0, u, u_hist)
            e = (yh - y) ** 2
            pen = l2 * jnp.sum(z[nfree - nres: nfree] ** 2) if nres > 0 else 0.0
            for (pn, pmu, psc) in struct.prior:
                pen = pen + struct.prior_w * ((th[pn] - pmu) / psc) ** 2
            return jnp.sum(w * e) / jnp.sum(w) + pen

        self.vg = jax.jit(jax.value_and_grad(loss_fn))
        self.loss = jax.jit(loss_fn)

        def sim_fn(z, u, u_hist):
            th, x0 = unpack(struct, z[:nfree], z[nfree:])
            return sim(th, x0, u, u_hist)[0]

        self.simj = jax.jit(sim_fn)

        # window fit of x0 only (theta frozen)
        def win_loss(zx, zf, u, y, u_hist):
            th, x0 = unpack(struct, zf, zx)
            yh, _ = sim(th, x0, u, u_hist)
            return jnp.mean((yh - y) ** 2)

        self.win_vg = jax.jit(jax.value_and_grad(win_loss))
        self.win_loss = jax.jit(win_loss)

    # -- parameter helpers -------------------------------------------------
    def init_z(self, rng: np.random.Generator, y0: float, jitter=0.35):
        zf = []
        for n in self.s.free:
            t, d = PARAMS[n]
            if n in ("d1", "d2"):
                zf.append(d + rng.normal(0, 1.0))
            elif t == "pos":
                zf.append(np.log(d) + rng.normal(0, jitter))
            else:
                scale = {"a": 0.8, "b4": 0.5, "c2": 0.01, "cA": 0.05, "c4": 0.02, "cA1": 0.05, "h0": 1.5}.get(n, 0.3)
                zf.append(d + rng.normal(0, scale))
        h = self.s.res_hidden
        if h > 0:
            zf += list(rng.normal(0, 0.5, 3 * h)) + [0.0] * h + list(rng.normal(0, 0.01, 2 * h)) + [0.0, 0.0]
        x2 = max(y0 + rng.normal(0, 0.3), 0.5)
        x1 = np.clip(rng.uniform(2, 9), 0.5, 9.9)
        return np.array(zf + [np.log(x1), np.log(x2)], float)

    def fit(self, u, y, w=None, n_starts=6, seed=0, maxiter=400, u_hist=None, verbose=False):
        u = np.asarray(u, float); y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        uh = np.array([u[0], u[0]]) if u_hist is None else np.asarray(u_hist, float)
        rng = np.random.default_rng(seed)
        best = None
        for k in range(n_starts):
            z0 = self.init_z(rng, y[0])
            try:
                res = minimize(lambda z: tuple(np.asarray(v, float) for v in self.vg(z, u, y, w, uh)),
                               z0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=maxiter * 2))
            except Exception as e:  # numerical failure in a start
                if verbose: print("start failed", e)
                continue
            if not np.isfinite(res.fun):
                continue
            if verbose: print(f"  start {k}: loss {res.fun:.5f} rmse {np.sqrt(res.fun):.4f} nit {res.nit}")
            if best is None or res.fun < best.fun:
                best = res
        return best.x, float(best.fun)

    def fit_x0(self, z, u_win, y_win, u_hist, n_starts=12, seed=0):
        """Window fit: x0 only, theta frozen at z[:nfree]. Grid + polish. Uses only the given window."""
        zf = jnp.asarray(z[: self.nfree])
        rng = np.random.default_rng(seed)
        best = None
        y0 = float(y_win[0])
        cands = []
        for x1 in (1.5, 3.0, 5.0, 7.0, 9.0):
            for dx in (-0.3, 0.3):
                cands.append(np.array([np.log(x1), np.log(max(y0 + dx - float(self._a(z)), 0.3))]))
        for z0 in cands[:n_starts]:
            res = minimize(lambda zx: tuple(np.asarray(v, float) for v in self.win_vg(zx, zf, u_win, y_win, u_hist)),
                           z0, jac=True, method="L-BFGS-B", options=dict(maxiter=100))
            if best is None or res.fun < best.fun:
                best = res
        return best.x, float(best.fun)

    def _a(self, z):
        if "a" in self.s.free:
            return float(z[self.s.free.index("a")])
        return PARAMS["a"][1]

    def simulate_with_init(self, z, zx, u, u_hist):
        zz = np.concatenate([np.asarray(z[: self.nfree]), np.asarray(zx)])
        return np.asarray(self.simj(zz, np.asarray(u, float), np.asarray(u_hist, float)))

    def theta(self, z):
        th, x0 = unpack(self.s, jnp.asarray(z[: self.nfree]), jnp.asarray(z[self.nfree:]))
        return {k: float(v) for k, v in th.items()}, np.asarray(x0)


def block_rmse(yh, y, n_init):
    e = np.asarray(yh)[n_init:] - np.asarray(y)[n_init:]
    return float(np.sqrt(np.mean(e ** 2)))


def lift_to_residual(z_small, struct_small: Structure, struct_big: Structure, rng: np.random.Generator, scale=0.5):
    """Warm start: copy the physics parameters/x0, add a near-zero residual MLP (output weights ~1e-2)."""
    assert struct_small.free == struct_big.free and struct_small.res_hidden == 0 and struct_big.res_hidden > 0
    nf = len(struct_small.free)
    h = struct_big.res_hidden
    res = list(rng.normal(0, scale, 3 * h)) + [0.0] * h + list(rng.normal(0, 0.01, 2 * h)) + [0.0, 0.0]
    return np.concatenate([np.asarray(z_small[:nf], float), np.array(res), np.asarray(z_small[nf:], float)])


def fit_two_stage(f_phys: "Fitter", f_res: "Fitter", u, y, w=None, n_starts=4, n_starts_res=3, seed=0, maxiter=400, maxiter_res=300):
    """Stage 1: physics-only multi-start. Stage 2: add residual MLP (L2-penalised), warm-started from stage 1."""
    z1, l1 = f_phys.fit(u, y, w=w, n_starts=n_starts, seed=seed, maxiter=maxiter)
    rng = np.random.default_rng(seed + 12345)
    uh = np.array([u[0], u[0]])
    w_ = np.ones_like(y) if w is None else w
    best = None
    for k in range(n_starts_res):
        z0 = lift_to_residual(z1, f_phys.s, f_res.s, rng)
        res = minimize(lambda z: tuple(np.asarray(v, float) for v in f_res.vg(z, u, y, w_, uh)),
                       z0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter_res, maxfun=2 * maxiter_res))
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    return best.x, float(best.fun), z1, l1
