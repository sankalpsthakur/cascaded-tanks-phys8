"""Masked-block cross-validation on the TRAINING record only (mirrors the official test protocol).

For each held-out block [s, e): fit theta on the whole-record simulation with the block's loss weights set to 0 (the
state runs through the block driven by u only), then score the block exactly like the official test: the state at the
block start is fitted ONLY from its first 50 samples with theta frozen, the block is free-run, RMSE over samples 50..end.
The official test record is never touched here.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402
from ct import phys  # noqa: E402

BLOCKS = [(0, 256), (256, 512), (512, 768), (768, 1024)]
N_INIT = 50


def cv_structure(struct: phys.Structure, u, y, n_starts=4, seed=0, maxiter=400, verbose=False):
    f = phys.Fitter(struct)
    out = []
    for bi, (s, e) in enumerate(BLOCKS):
        w = np.ones(len(y)); w[s:e] = 0.0
        t0 = time.time()
        z, loss = f.fit(u, y, w=w, n_starts=n_starts, seed=seed * 100 + bi, maxiter=maxiter)
        # score the held-out block
        ub, yb = u[s:e], y[s:e]
        uh = u[s - 2:s] if s >= 2 else np.array([u[0], u[0]])
        zx, wl = f.fit_x0(z, ub[:N_INIT], yb[:N_INIT], uh)
        yh = f.simulate_with_init(z, zx, ub, uh)
        rm = phys.block_rmse(yh, yb, N_INIT)
        rm_all = phys.block_rmse(yh, yb, 0)
        out.append(dict(block=bi, train_rmse=float(np.sqrt(loss)), val_rmse=rm, val_rmse_incl_window=rm_all,
                        window_fit_rmse=float(np.sqrt(wl)), secs=time.time() - t0))
        if verbose:
            print(f"  fold {bi} train {np.sqrt(loss):.4f} val {rm:.4f} win {np.sqrt(wl):.4f} ({time.time()-t0:.0f}s)", flush=True)
    se = np.array([o["val_rmse"] ** 2 for o in out])
    pooled = float(np.sqrt(se.mean()))
    return dict(name=struct.name, free=list(struct.free), tie_r2=struct.tie_r2, clamp_x2=struct.clamp_x2, nsub=struct.nsub,
                folds=out, cv_rmse=pooled, cv_fold_rmse=[o["val_rmse"] for o in out],
                n_params=len(struct.free) + 2)


if __name__ == "__main__":
    u, y, _ = P.load_train()
    base = ("k1", "r2", "k3", "k4", "beta", "a")
    ladder = {
        "A  L1 base":        phys.Structure(free=base, tie_r2=False),
        "B  +b4":            phys.Structure(free=base + ("b4",), tie_r2=False),
        "C  +b4+cA":         phys.Structure(free=base + ("b4", "cA"), tie_r2=False),
        "D  +b4+cA+c2":      phys.Structure(free=base + ("b4", "cA", "c2"), tie_r2=False),
        "E  +b4+p1+p3":      phys.Structure(free=base + ("b4", "p1", "p3"), tie_r2=False),
        "F  +b4+fir":        phys.Structure(free=base + ("b4", "d1", "d2"), tie_r2=False),
        "G  L0 worden-like": phys.Structure(free=("k1", "k3", "k4", "beta"), tie_r2=True),
        "H  base noclamp2":  phys.Structure(free=base, tie_r2=False, clamp_x2=False),
    }
    sel = sys.argv[1:]  # optional name prefixes
    outp = os.path.join(ROOT, "results", "cv_ladder.jsonl")
    for name, st in ladder.items():
        if sel and not any(name.startswith(x) for x in sel):
            continue
        st = phys.Structure(**{**st.__dict__, "name": name})
        t0 = time.time()
        print(f"== {name}", flush=True)
        r = cv_structure(st, u, y, verbose=True)
        r["secs"] = time.time() - t0
        print(f"   CV pooled RMSE {r['cv_rmse']:.4f}  folds {np.round(r['cv_fold_rmse'],3)}  ({r['secs']:.0f}s)", flush=True)
        with open(outp, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def cv_generic(name, make_fit, u, y, verbose=True):
    """Generic masked-block CV. make_fit(u, y, w, seed) -> (fitter, z) where fitter exposes fit_x0/simulate_with_init."""
    out = []
    for bi, (s, e) in enumerate(BLOCKS):
        w = np.ones(len(y)); w[s:e] = 0.0
        t0 = time.time()
        fitter, z, extra = make_fit(u, y, w, bi)
        ub, yb = u[s:e], y[s:e]
        uh = u[s - 2:s] if s >= 2 else np.array([u[0], u[0]])
        zx, wl = fitter.fit_x0(z, ub[:N_INIT], yb[:N_INIT], uh)
        yh = fitter.simulate_with_init(z, zx, ub, uh)
        rm = phys.block_rmse(yh, yb, N_INIT)
        try:
            zsave = [np.asarray(zz).tolist() for zz in fitter.zs] if hasattr(fitter, "zs") else np.asarray(z).tolist()
        except Exception:
            zsave = None
        out.append(dict(block=bi, val_rmse=rm, window_fit_rmse=float(np.sqrt(wl)), secs=time.time() - t0, z=zsave, **extra))
        if verbose:
            print(f"  fold {bi} val {rm:.4f} win {np.sqrt(wl):.4f} { {k: v for k, v in extra.items()} } ({time.time()-t0:.0f}s)", flush=True)
    se = np.array([o["val_rmse"] ** 2 for o in out])
    return dict(name=name, folds=out, cv_rmse=float(np.sqrt(se.mean())), cv_fold_rmse=[o["val_rmse"] for o in out])
