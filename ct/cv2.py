"""Second CV round (train-only, masked blocks): priors on exponents, residual MLP, input FIR, bagging. Appends to results/cv_round2.jsonl"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P, phys  # noqa: E402
from ct.cv import cv_generic, BLOCKS, N_INIT  # noqa: E402

BASE = ("k1", "r2", "k3", "k4", "beta", "a", "b4", "cA")


def single(struct, n_starts=4):
    f = phys.Fitter(struct)

    def make_fit(u, y, w, bi):
        z, loss = f.fit(u, y, w=w, n_starts=n_starts, seed=bi * 100, maxiter=400)
        return f, z, dict(train_rmse=float(np.sqrt(loss)))
    return make_fit


def two_stage(s_phys, s_res, n_starts=3, n_res=3):
    fp, fr = phys.Fitter(s_phys), phys.Fitter(s_res)

    def make_fit(u, y, w, bi):
        z, loss, z1, l1 = phys.fit_two_stage(fp, fr, u, y, w=w, n_starts=n_starts, n_starts_res=n_res, seed=bi * 100)
        return fr, z, dict(train_rmse=float(np.sqrt(max(loss, 0))), phys_train_rmse=float(np.sqrt(l1)))
    return make_fit


class Bag:
    """Ensemble wrapper exposing the fit_x0 / simulate_with_init interface; mean of member simulations."""
    def __init__(self, fitter, zs):
        self.f, self.zs = fitter, zs

    def fit_x0(self, z, u_win, y_win, u_hist):
        zxs = [self.f.fit_x0(zz, u_win, y_win, u_hist)[0] for zz in self.zs]
        wl = np.mean([self.f.fit_x0(zz, u_win, y_win, u_hist)[1] for zz in self.zs])
        return zxs, float(wl)

    def simulate_with_init(self, z, zxs, u, u_hist):
        return np.mean([self.f.simulate_with_init(zz, zx, u, u_hist) for zz, zx in zip(self.zs, zxs)], axis=0)


def bagged(struct, K=4, drop=0.2, blk=32, n_starts=2):
    f = phys.Fitter(struct)

    def make_fit(u, y, w, bi):
        zs, tr = [], []
        for m in range(K):
            rng = np.random.default_rng(1000 * bi + m)
            wm = w.copy()
            nblk = len(y) // blk
            drop_blocks = rng.choice(nblk, int(drop * nblk), replace=False)
            for b in drop_blocks:
                wm[b * blk:(b + 1) * blk] = 0.0
            z, loss = f.fit(u, y, w=wm, n_starts=n_starts, seed=bi * 100 + m, maxiter=400)
            zs.append(z); tr.append(np.sqrt(loss))
        return Bag(f, zs), zs[0], dict(train_rmse=float(np.mean(tr)), K=K)
    return make_fit


if __name__ == "__main__":
    u, y, _ = P.load_train()
    pri = (("p1", 0.5, 0.1), ("p3", 0.5, 0.1))
    cand = {
        "C0 base(+b4+cA)": lambda: single(phys.Structure(free=BASE, tie_r2=False)),
        "P1 +p1p3 prior.1 w1e-3": lambda: single(phys.Structure(free=BASE + ("p1", "p3"), tie_r2=False, prior=pri, prior_w=1e-3)),
        "P2 +p1p3 prior.1 w1e-2": lambda: single(phys.Structure(free=BASE + ("p1", "p3"), tie_r2=False, prior=pri, prior_w=1e-2)),
        "R1 res4 l2=1e-3": lambda: two_stage(phys.Structure(free=BASE, tie_r2=False), phys.Structure(free=BASE, tie_r2=False, res_hidden=4, res_l2=1e-3)),
        "R2 res4 l2=1e-2": lambda: two_stage(phys.Structure(free=BASE, tie_r2=False), phys.Structure(free=BASE, tie_r2=False, res_hidden=4, res_l2=1e-2)),
        "F1 +fir": lambda: single(phys.Structure(free=BASE + ("d1", "d2"), tie_r2=False)),
        "Q1 +c4 (pump curvature)": lambda: single(phys.Structure(free=BASE + ("c4",), tie_r2=False)),
        "Q2 +cA1 (upper capacity)": lambda: single(phys.Structure(free=BASE + ("cA1",), tie_r2=False)),
        "Q3 +c4+cA1": lambda: single(phys.Structure(free=BASE + ("c4", "cA1"), tie_r2=False)),
        "O1 +qov (overflow routing exponent)": lambda: single(phys.Structure(free=BASE + ("qov",), tie_r2=False)),
        "O2 +qov+c4": lambda: single(phys.Structure(free=BASE + ("qov", "c4"), tie_r2=False)),
        "H1 +h0 (upper outlet height)": lambda: single(phys.Structure(free=BASE + ("h0",), tie_r2=False)),
        "H2 +h0+qov": lambda: single(phys.Structure(free=BASE + ("h0", "qov"), tie_r2=False)),
        "Z1 deadzone pump": lambda: single(phys.Structure(free=BASE, tie_r2=False, deadzone=True)),
        "Z2 deadzone +c4": lambda: single(phys.Structure(free=BASE + ("c4",), tie_r2=False, deadzone=True)),
        "Z3 deadzone +h0": lambda: single(phys.Structure(free=BASE + ("h0",), tie_r2=False, deadzone=True)),
        "B1 bag4 of C0": lambda: bagged(phys.Structure(free=BASE, tie_r2=False)),
    }
    sel = sys.argv[1:]
    for name, mk in cand.items():
        if sel and not any(name.startswith(x) for x in sel):
            continue
        t0 = time.time()
        print("==", name, flush=True)
        r = cv_generic(name, mk(), u, y)
        r["secs"] = time.time() - t0
        print(f"   CV pooled RMSE {r['cv_rmse']:.4f} folds {np.round(r['cv_fold_rmse'],3)} ({r['secs']:.0f}s)", flush=True)
        with open(os.path.join(ROOT, "results", "cv_round2.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")
