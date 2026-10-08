"""FROZEN-configuration run. Reads a committed FREEZE json, refuses to run on a dirty git tree, trains on the full
training record for each seed, then scores each seed ONCE on the official test record via protocol.test_eval (which logs
every evaluation in TEST-LOG.md). Nothing here selects anything; the config is fixed before this script runs.

    python ct/final.py results/FREEZE_ct_<name>.json --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402
from ct import phys  # noqa: E402


def moving_block_bootstrap_ci(resid, block=25, n_boot=4000, seed=0, alpha=0.05):
    """CI for RMSE of a (serially correlated) residual series via moving-block bootstrap."""
    r2 = np.asarray(resid, float) ** 2
    n = len(r2)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        stats[i] = np.sqrt(r2[idx].mean())
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def tree_clean_and_committed(freeze_path):
    # only code + freeze files must be clean/committed (logs and the other track's results change concurrently)
    st = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--", "ct", "results/FREEZE_*.json"], text=True).strip()
    rel = os.path.relpath(os.path.abspath(freeze_path), ROOT)
    tracked = subprocess.call(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    return st == "", tracked, st


def build_structure(cfg):
    s = cfg["structure"]
    return phys.Structure(free=tuple(s["free"]), tie_r2=s.get("tie_r2", True), clamp_x2=s.get("clamp_x2", True),
                          nsub=s.get("nsub", 4), res_hidden=s.get("res_hidden", 0), res_l2=s.get("res_l2", 1e-3), name=s.get("name", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freeze")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="dry run on TRAIN only; never scores the test record")
    a = ap.parse_args()
    cfg = json.load(open(a.freeze))
    clean, tracked, st = tree_clean_and_committed(a.freeze)
    if not a.allow_dirty and not (clean and tracked):
        sys.exit(f"REFUSING to touch the test record: tree clean={clean}, freeze tracked={tracked}\n{st}")
    struct = build_structure(cfg)
    fit = cfg["fit"]
    u, y, _ = P.load_train()
    fitter = phys.Fitter(struct)
    out = dict(freeze=os.path.relpath(a.freeze, ROOT), code=P.git_head(), seeds={}, config_id=P.config_id(cfg))
    for seed in a.seeds:
        t0 = time.time()
        z, loss = fitter.fit(u, y, n_starts=fit["n_starts"], seed=seed, maxiter=fit["maxiter"])
        train_rmse = float(np.sqrt(loss))
        th, x0 = fitter.theta(z)
        fit_secs = time.time() - t0

        def simulate_fn(u_test, y_init, z=z):
            uh = np.array([u_test[0], u_test[0]])
            zx, _ = fitter.fit_x0(z, u_test[:P.N_INIT], y_init, uh)
            return fitter.simulate_with_init(z, zx, u_test, uh)

        if a.allow_dirty:
            print("dry run (train only): seed", seed, "train RMSE", round(train_rmse, 4)); continue
        rmse, ymod = P.test_eval(simulate_fn, tag=f"FROZEN {cfg['name']}", config=cfg, seed=seed,
                                 purpose="frozen-config final run (one score per seed)",
                                 extra=f"train_rmse={train_rmse:.4f}; fit={fit_secs:.0f}s")
        _, test = P._load_both()
        yt = np.asarray(test.y, float).ravel()
        resid = (ymod - yt)[P.N_INIT:]
        lo, hi = moving_block_bootstrap_ci(resid)
        out["seeds"][str(seed)] = dict(test_rmse=rmse, train_rmse=train_rmse, theta=th, ci95_block_bootstrap=[lo, hi], fit_secs=fit_secs)
        np.save(os.path.join(ROOT, "runs", f"ct_final_{cfg['name']}_seed{seed}_ypred.npy"), ymod)
        print(f"seed {seed}: test RMSE {rmse:.4f}  train {train_rmse:.4f}  95% block-bootstrap CI [{lo:.3f}, {hi:.3f}]", flush=True)
    if out["seeds"]:
        v = np.array([d["test_rmse"] for d in out["seeds"].values()])
        out["summary"] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, min=float(v.min()), max=float(v.max()))
        json.dump(out, open(os.path.join(ROOT, "results", f"ct_final_{cfg['name']}.json"), "w"), indent=1)
        print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
