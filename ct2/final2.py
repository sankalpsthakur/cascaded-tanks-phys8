"""Round-2 FROZEN-configuration run. Reads a committed FREEZE json, refuses to run on a dirty tree (ct/, ct2/, freeze files, pre-registration),
fits every seed on the TRAIN record in a worker pool, checks the train-RMSE gate, and only then scores each seed ONCE on the official test
record in the parent process. Every test scoring is a row in TEST-LOG-CT2.md (the kernel's copy; ingested into TEST-LOG.md by tools/ct2_ingest_testlog.py).
Output files have the layout of ct/final.py so that ct/compare_ref.py, ct/plots.py and tools/ct_aggregate.py work unchanged.

    python ct2/final2.py results/FREEZE_ct_round2.json --seeds 0 1 2 3 4 5 6 7 8 9 --workers 4
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, datetime as dt, fcntl, json, multiprocessing as mp, os, subprocess, sys, time, traceback
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402
from ct.final import moving_block_bootstrap_ci  # noqa: E402  (round-1 module, unchanged)
from ct2 import phys2  # noqa: E402

LOG2 = os.path.join(ROOT, "TEST-LOG-CT2.md")
FIRST_ROW = 25      # TEST-LOG.md rows 1-24 belong to round 1 (reference-code runs, disclosure, ten frozen ct_phys8 scores)
GUARD_PATHS = ["ct", "ct2", "results/FREEZE_*.json", "PREREGISTRATION-CT-ROUND2.md"]

_HEADER = (
    "## CT round 2: frozen `ct_round2` run (kernel copy; rows are numbered from 25 to continue the Cascaded Tanks rows of TEST-LOG.md)\n\n"
    "Official metric `nonlinear_benchmarks` RMSE(y_test, y_model, n_init=50), free-run simulation, volts. One row per scoring of the test record, failures included.\n\n"
    "| # | UTC | tag | config | seed | RMSE n_init=50 | extra | purpose | code @ |\n"
    "|---|-----|-----|--------|------|----------------|-------|---------|--------|\n"
)


def git_head():
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--"] + GUARD_PATHS, text=True, stderr=subprocess.DEVNULL).strip())
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"


def append_row(tag, config_id, seed, rmse, extra, purpose):
    with open(os.path.join(ROOT, ".testlog.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(LOG2):
            open(LOG2, "w").write(_HEADER)
        n = FIRST_ROW + sum(1 for l in open(LOG2).read().splitlines() if l.startswith("| ") and l[2:3].isdigit())
        line = "| %d | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
            n, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), tag, config_id, seed, rmse, str(extra).replace("|", "/"), purpose, git_head())
        with open(LOG2, "a") as f:
            f.write(line)
        return n


def tree_clean_and_committed(freeze_path):
    st = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--"] + GUARD_PATHS, text=True).strip()
    rel = os.path.relpath(os.path.abspath(freeze_path), ROOT)
    tracked = subprocess.call(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    return st == "", tracked, st


def fitter_from(cfg):
    s, f = cfg["structure"], cfg["fit"]
    struct = phys2.Structure(free=tuple(s["free"]), clamp_x2=s.get("clamp_x2", True), nsub=s.get("nsub", 4), name=s.get("name", ""))
    kw = {k: f[k] for k in ("loss", "huber_delta", "ms_K", "ms_lam") if k in f}
    return phys2.Fitter(struct, **kw)


_W = {}


def _train_only():
    """fit workers load the estimation record only (load_train returns the train arrays)"""
    if "u" not in _W:
        u, y, _ = P.load_train()
        _W["u"], _W["y"] = u, y
    return _W["u"], _W["y"]


def fit_seed(cfg, seed):
    """TRAIN-only fit of one seed. Returns z and diagnostics (picklable)."""
    t0 = time.time()
    u, y = _train_only()
    if "f" not in _W:
        _W["f"] = fitter_from(cfg)
    f = _W["f"]
    fit = cfg["fit"]
    z, loss, diag = f.fit(u, y, n_starts=fit["n_starts"], seed=seed, maxiter=fit["maxiter"])
    th, x0 = f.theta(z)
    uh = np.array([u[0], u[0]])
    yh = f.simulate_with_init(z, np.log(x0), u, uh)
    return dict(seed=seed, z=[float(a) for a in z], objective=float(loss), train_rmse=float(np.sqrt(np.mean((yh - y) ** 2))),
                theta=th, starts=[{k: v for k, v in d.items() if k != "err"} for d in diag], fit_secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freeze")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4, help="0 = fit in this process (local plumbing tests)")
    ap.add_argument("--allow-dirty", action="store_true", help="train-only dry run; never scores the test record")
    a = ap.parse_args()
    cfg = json.load(open(a.freeze))
    if cfg.get("identical_to_ct_phys8"):
        sys.exit("this freeze is identical to ct_phys8: no new test scoring is allowed (TEST-LOG.md rows 15-24 stand)")
    clean, tracked, st = tree_clean_and_committed(a.freeze)
    if not a.allow_dirty and not (clean and tracked):
        sys.exit(f"REFUSING to touch the test record: tree clean={clean}, freeze tracked={tracked}\n{st}")
    cid = P.config_id(cfg)
    out = dict(freeze=os.path.relpath(a.freeze, ROOT), code=git_head(), seeds={}, config_id=cid)

    # ---- phase A: train-only fits, in parallel ----------------------------------------------------------------------
    t0 = time.time()
    fits = {}

    def got(r):
        fits[r["seed"]] = r
        print(f"fit seed {r['seed']}: train RMSE {r['train_rmse']:.5f}  objective {r['objective']:.6f}  {r['fit_secs']:.0f}s  (elapsed {time.time() - t0:.0f}s)", flush=True)

    if a.workers == 0:          # in-process (local plumbing tests only)
        for s in a.seeds:
            got(fit_seed(cfg, s))
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(fit_seed, cfg, s): s for s in a.seeds}
            for fu in cf.as_completed(futs):
                got(fu.result())

    # ---- gate: the train RMSE of seed 0 must reproduce the selection kernel's dry run --------------------------------
    gate = cfg.get("gate")
    if gate and 0 in fits:
        d = abs(fits[0]["train_rmse"] - gate["train_rmse_seed0"])
        ok = d <= gate["tol"]
        print(f"GATE train RMSE seed 0 = {fits[0]['train_rmse']:.5f}, dry run {gate['train_rmse_seed0']:.5f} +- {gate['tol']} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            sys.exit(f"gate failed; the test record was not touched")
    if a.allow_dirty:
        print("dry run (train only) done; test record untouched")
        return

    # ---- phase B: score every seed once on the test record (parent process) ------------------------------------------
    f = fitter_from(cfg)
    for seed in a.seeds:
        r = fits[seed]
        z = np.array(r["z"])

        def simulate_fn(u_test, y_init, z=z):
            uh = np.array([u_test[0], u_test[0]])
            zx, _ = f.fit_x0(z, u_test[:P.N_INIT], y_init, uh)
            return f.simulate_with_init(z, zx, u_test, uh)

        try:
            rmse, ymod = P.test_eval(simulate_fn, tag=f"FROZEN {cfg['name']}", config=cfg, seed=seed, purpose="", log=False)
        except Exception:
            append_row(f"FROZEN {cfg['name']} - CRASHED", cid, seed, "n/a (crash)", traceback.format_exc()[-200:].replace("\n", " "), "frozen-config final run (one score per seed); failure logged")
            raise
        append_row(f"FROZEN {cfg['name']}", cid, seed, f"{rmse:.4f}", f"train_rmse={r['train_rmse']:.4f}; fit={r['fit_secs']:.0f}s", "frozen-config final run (one score per seed)")
        _, test = P._load_both()
        yt = np.asarray(test.y, float).ravel()
        resid = (ymod - yt)[P.N_INIT:]
        lo, hi = moving_block_bootstrap_ci(resid)
        out["seeds"][str(seed)] = dict(test_rmse=rmse, train_rmse=r["train_rmse"], theta=r["theta"], ci95_block_bootstrap=[lo, hi], fit_secs=r["fit_secs"], starts=r["starts"], z=r["z"])
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        np.save(os.path.join(ROOT, "runs", f"ct_final_{cfg['name']}_seed{seed}_ypred.npy"), ymod)
        print(f"seed {seed}: test RMSE {rmse:.4f}  train {r['train_rmse']:.4f}  95% block-bootstrap CI [{lo:.3f}, {hi:.3f}]", flush=True)
    v = np.array([d["test_rmse"] for d in out["seeds"].values()])
    out["summary"] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, min=float(v.min()), max=float(v.max()))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"ct_final_{cfg['name']}.json"), "w"), indent=1)
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
