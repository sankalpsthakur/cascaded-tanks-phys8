"""Level-extrapolation validation on the TRAINING record only: mask (zero loss weight) the lowest-y samples and score the
model on exactly those samples. The official test reaches lower levels than the estimation record (min y 2.16 vs 2.91), which
block-CV cannot probe. Also prints the fitted sensor offset `a` and which basin each structure lands in."""
from __future__ import annotations
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P, phys  # noqa: E402

BASE = ("k1", "r2", "k3", "k4", "beta", "a", "b4", "cA")


def level_cv(struct, u, y, q=0.12, n_starts=6, seed=0, force_a_sign=None):
    thr = np.quantile(y, q)
    w = (y >= thr).astype(float)
    # do not mask the first samples (initial state) -> keep x0 identifiable
    w[:50] = 1.0
    f = phys.Fitter(struct)
    z, loss = f.fit(u, y, w=w, n_starts=n_starts, seed=seed)
    th, x0 = f.theta(z)
    uh = np.array([u[0], u[0]])
    yh = f.simulate_with_init(z, np.log(x0), u, uh)
    m = (w == 0)
    rm_masked = float(np.sqrt(np.mean((yh[m] - y[m]) ** 2)))
    rm_unmasked = float(np.sqrt(np.mean((yh[~m] - y[~m]) ** 2)))
    return dict(name=struct.name, thr=float(thr), n_masked=int(m.sum()), rmse_masked=rm_masked, rmse_unmasked=rm_unmasked, a=th["a"], theta=th)


if __name__ == "__main__":
    u, y, _ = P.load_train()
    cands = {
        "A  base": phys.Structure(free=("k1", "r2", "k3", "k4", "beta", "a"), tie_r2=False),
        "B  +b4": phys.Structure(free=("k1", "r2", "k3", "k4", "beta", "a", "b4"), tie_r2=False),
        "C  +b4+cA": phys.Structure(free=BASE, tie_r2=False),
        "Z1 deadzone": phys.Structure(free=BASE, tie_r2=False, deadzone=True),
        "E  +p1+p3": phys.Structure(free=BASE + ("p1", "p3"), tie_r2=False),
    }
    sel = sys.argv[1:]
    for name, st in cands.items():
        if sel and not any(name.startswith(x) for x in sel):
            continue
        st = phys.Structure(**{**st.__dict__, "name": name})
        t0 = time.time()
        r = level_cv(st, u, y)
        print(f"{name:14s} masked(n={r['n_masked']}, y<{r['thr']:.2f}) RMSE {r['rmse_masked']:.3f} | unmasked {r['rmse_unmasked']:.3f} | a={r['a']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(os.path.join(ROOT, "results", "cv_level.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")
