"""Round-2 selection driver for Cascaded Tanks (ESTIMATION record only; see PREREGISTRATION-CT-ROUND2.md).

Stage 1: every candidate structure with the baseline fit G0, four masked-block CV jobs + one level-extrapolation job each.
Rule 1 : smallest structure within one standard error of the best pooled CV RMSE; ties -> level-extrapolation RMSE.
Stage 2: fit variants G3 (more starts), G1 (pseudo-Huber), G2 (multiple shooting) on the structure chosen by rule 1.
Rule 2 : same rule among the fit variants (G0 from stage 1 included).
Dry run: the chosen configuration is fitted on the full estimation record with the final hyper-parameters (seed 0, TRAIN only).
Nothing here can score the test record: the loader is patched to return (train, None) and test_eval / log_external raise.

usage: python ct2/select2.py --out DIR [--workers 4] [--deadline-min 110] [--only S00,S01] [--maxiter-scale 1.0]
"""
from __future__ import annotations

import argparse, json, os, sys, time
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402

# ---- the selection code must not be able to read or score the test record -------------------------------------------
def _no_test(*a, **k):
    raise RuntimeError("round-2 selection must not touch the test record")


_orig_load_both = P._load_both
def _train_only():
    train, _ = _orig_load_both()
    return train, None


P.test_eval = _no_test
P.log_external = _no_test
P._load_both = _train_only

from ct2 import phys2  # noqa: E402

BLOCKS = [(0, 256), (256, 512), (512, 768), (768, 1024)]
N_INIT = 50
LEVEL_Q = 0.12
BASE = ("k1", "r2", "k3", "k4", "beta", "a", "b4", "cA")
SWAP = tuple(n for n in BASE if n != "cA") + ("w3",)
CV_STARTS, CV_MAXITER, LEVEL_STARTS = 4, 400, 6
FINAL_MAXITER = 600
GATE_CV_C, GATE_CV_TOL = 0.2089, 0.003      # reproduction gate on S00 (round-1 pooled CV)
SAME_LEVEL_TOL = 0.005                       # level-extrapolation RMSEs closer than this count as equal
IDENT_REL_TOL, IDENT_TRAIN_TOL = 1e-3, 5e-4  # "identical to ct_phys8" test on the dry run

#            id     label                                  free (BASE + extras)           nested in S00
CANDS = [
    ("S00", "C = ct_phys8 (reference)",                    BASE,                            False),
    ("S01", "+tp pump/inflow lag",                         BASE + ("tp",),                  True),
    ("S02", "+w3 level-dependent lower outlet",            BASE + ("w3",),                  True),
    ("S03", "+to delayed overflow return",                 BASE + ("to",),                  True),
    ("S04", "+qov flow-dependent overflow routing",        BASE + ("qov",),                 True),
    ("S05", "+c4 pump curvature",                          BASE + ("c4",),                  True),
    ("S06", "+h0 upper outlet height (round-1 H1)",        BASE + ("h0",),                  True),
    ("S07", "+FIR input taps (round-1 F1)",                BASE + ("d1", "d2"),             True),
    ("S08", "+cA1 upper-tank capacity slope",              BASE + ("cA1",),                 True),
    ("S09", "+tp +w3",                                     BASE + ("tp", "w3"),             True),
    ("S10", "+tp +to",                                     BASE + ("tp", "to"),             True),
    ("S11", "+qov +to",                                    BASE + ("qov", "to"),            True),
    ("S12", "+tp +w3 +to",                                 BASE + ("tp", "w3", "to"),       True),
    ("S13", "swap: cA replaced by w3",                     SWAP,                            False),
]
CAND = {c[0]: c for c in CANDS}
# fit variants; the order is the tie preference (simplest first)
VARIANTS = {
    "G0": dict(label="baseline: MSE, single shooting, 4 starts", loss="mse", ms_K=0, n_starts=4, final_starts=8),
    "G3": dict(label="more random starts (8 in CV, 16 in the final fit)", loss="mse", ms_K=0, n_starts=8, final_starts=16),
    "G1": dict(label="pseudo-Huber loss, delta 0.2 V", loss="huber", huber_delta=0.2, ms_K=0, n_starts=4, final_starts=8),
    "G2": dict(label="multiple shooting, 8 segments of 128, lambda 30", loss="mse", ms_K=8, ms_lam=30.0, n_starts=4, final_starts=8),
}
VARIANT_ORDER = ["G0", "G3", "G1", "G2"]


def now():
    return time.strftime("%H:%M:%S", time.gmtime())


# ---- worker side ------------------------------------------------------------------------------------------------------
_DATA = {}
_FIT = {}


def _data():
    if "u" not in _DATA:
        u, y, _ = P.load_train()
        _DATA["u"], _DATA["y"] = u, y
    return _DATA["u"], _DATA["y"]


def make_fitter(cand_id, variant_id):
    key = (cand_id, variant_id)
    if key not in _FIT:
        if len(_FIT) >= int(os.environ.get("CT2_FIT_CACHE", "3")):
            _FIT.pop(next(iter(_FIT)))
            import jax
            jax.clear_caches()
        v = VARIANTS[variant_id]
        kw = {k: v[k] for k in ("loss", "huber_delta", "ms_K", "ms_lam") if k in v}
        _FIT[key] = phys2.Fitter(phys2.Structure(free=CAND[cand_id][2], name=cand_id), **kw)
    return _FIT[key]


def warm_from_c(cand_id, zc):
    """z of S00 (8 base numbers + 2 log-states) -> start vector of a nested candidate with the new terms switched off."""
    assert CAND[cand_id][3], cand_id + " is not nested in S00"
    zbase = dict(zip(phys2.Structure(free=BASE).free, np.asarray(zc[:len(BASE)], float)))
    free = phys2.Structure(free=CAND[cand_id][2]).free          # canonical (PARAMS) order = layout of z
    return np.concatenate([[zbase[n] if n in zbase else phys2.NEUTRAL_Z[n] for n in free], np.asarray(zc[len(BASE):], float)])


def run_job(spec):
    """spec: cand, variant, job ('fold0'..'fold3' | 'level'), warm (list of z lists), deadline (epoch s), maxiter_scale."""
    t0 = time.time()
    cand_id, vid, job = spec["cand"], spec["variant"], spec["job"]
    if t0 > spec["deadline"]:
        return dict(cand=cand_id, variant=vid, job=job, skipped=True, reason="deadline")
    try:
        u, y = _data()
        f = make_fitter(cand_id, vid)
        v = VARIANTS[vid]
        warm = [np.asarray(z, float) for z in spec.get("warm") or []]
        sc = spec.get("maxiter_scale", 1.0)
        out = dict(cand=cand_id, variant=vid, job=job, n_numbers=len(CAND[cand_id][2]) + 2)
        if job.startswith("fold"):
            bi = int(job[4:])
            s, e = BLOCKS[bi]
            w = np.ones(len(y)); w[s:e] = 0.0
            z, loss, diag = f.fit(u, y, w=w, n_starts=v["n_starts"], seed=bi * 100, maxiter=max(2, int(CV_MAXITER * sc)), warm=warm)
            ub, yb = u[s:e], y[s:e]
            uh = u[s - 2:s] if s >= 2 else np.array([u[0], u[0]])
            zx, wl = f.fit_x0(z, ub[:N_INIT], yb[:N_INIT], uh)
            yh = f.simulate_with_init(z, zx, ub, uh)
            out.update(val_rmse=phys2.block_rmse(yh, yb, N_INIT), window_fit_rmse=float(np.sqrt(wl)))
        elif job == "level":
            thr = float(np.quantile(y, LEVEL_Q))
            w = (y >= thr).astype(float); w[:N_INIT] = 1.0
            z, loss, diag = f.fit(u, y, w=w, n_starts=LEVEL_STARTS, seed=0, maxiter=max(2, int(CV_MAXITER * sc)), warm=warm)
            th, x0 = f.theta(z)
            uh = np.array([u[0], u[0]])
            yh = f.simulate_with_init(z, np.log(x0), u, uh)
            m = (w == 0)
            out.update(thr=thr, n_masked=int(m.sum()), rmse_masked=float(np.sqrt(np.mean((yh[m] - y[m]) ** 2))),
                       rmse_unmasked=float(np.sqrt(np.mean((yh[~m] - y[~m]) ** 2))))
        else:
            raise ValueError(job)
        th, x0 = f.theta(z)
        out.update(train_loss=loss, z=[float(a) for a in z], theta={k: th[k] for k in CAND[cand_id][2]}, starts=diag,
                   secs=round(time.time() - t0, 1), pid=os.getpid())
        return out
    except Exception as ex:  # one failing job must not kill the run
        import traceback
        return dict(cand=cand_id, variant=vid, job=job, error=repr(ex)[:300], tb=traceback.format_exc()[-1200:], secs=round(time.time() - t0, 1))


def dry_job(spec):
    """Full-record fit of the chosen configuration with the final hyper-parameters (TRAIN only)."""
    t0 = time.time()
    u, y = _data()
    cand_id, vid = spec["cand"], spec["variant"]
    f = make_fitter(cand_id, vid)
    v = VARIANTS[vid]
    z, loss, diag = f.fit(u, y, n_starts=v["final_starts"], seed=spec.get("seed", 0), maxiter=int(FINAL_MAXITER * spec.get("maxiter_scale", 1.0)))
    th, x0 = f.theta(z)
    uh = np.array([u[0], u[0]])
    yh = f.simulate_with_init(z, np.log(x0), u, uh)
    return dict(cand=cand_id, variant=vid, train_rmse=float(np.sqrt(np.mean((yh - y) ** 2))), objective=loss,
                theta={k: th[k] for k in CAND[cand_id][2]}, x0=[float(a) for a in x0], starts=diag, secs=round(time.time() - t0, 1))


# ---- driver side ------------------------------------------------------------------------------------------------------
def pooled_and_se(folds):
    m = np.asarray(folds, float) ** 2
    pooled = float(np.sqrt(m.mean()))
    se = float(np.std(m, ddof=1) / np.sqrt(len(m)) / (2.0 * pooled))
    return pooled, se


def table(results, cands, variant):
    """Per-candidate summary from job results; a candidate is complete only if all 5 jobs finished with finite numbers."""
    rows = {}
    for cid in cands:
        folds, lvl, tsec, err = [None] * 4, None, 0.0, None
        for bi in range(4):
            r = results.get((cid, variant, f"fold{bi}"))
            if r is None or "val_rmse" not in r or not np.isfinite(r["val_rmse"]):
                err = err or (r.get("error") or r.get("reason") or "missing") if r else "missing"
            else:
                folds[bi] = r["val_rmse"]; tsec += r["secs"]
        r = results.get((cid, variant, "level"))
        if r is None or "rmse_masked" not in r or not np.isfinite(r["rmse_masked"]):
            err = err or (r.get("error") or r.get("reason") or "missing") if r else "missing"
        else:
            lvl = r["rmse_masked"]; tsec += r["secs"]
        row = dict(cand=cid, variant=variant, label=CAND[cid][1], free=list(CAND[cid][2]), n_numbers=len(CAND[cid][2]) + 2,
                   folds=folds, level_rmse=lvl, secs=round(tsec, 1), complete=err is None)
        if err is None:
            row["cv"], row["se"] = pooled_and_se(folds)
        else:
            row["error"] = str(err)[:200]
        rows[cid] = row
    return rows


def apply_rule(rows, order):
    """rows: {key: row} with cv/se/level_rmse/n_numbers. order: list of keys giving the tie preference. Returns (chosen key, details)."""
    comp = {k: r for k, r in rows.items() if r["complete"]}
    if not comp:
        return None, dict(error="no complete candidate")
    best = min(comp, key=lambda k: comp[k]["cv"])
    thr = comp[best]["cv"] + comp[best]["se"]
    elig = [k for k in comp if comp[k]["cv"] <= thr]
    nmin = min(comp[k]["n_numbers"] for k in elig)
    ties = [k for k in elig if comp[k]["n_numbers"] == nmin]
    lv = min(comp[k]["level_rmse"] for k in ties)
    near = [k for k in ties if comp[k]["level_rmse"] <= lv + SAME_LEVEL_TOL]
    chosen = sorted(near, key=lambda k: order.index(k))[0]
    return chosen, dict(best=best, best_cv=comp[best]["cv"], se_best=comp[best]["se"], threshold=thr, eligible=elig, n_numbers_min=nmin,
                        tied_on_size=ties, best_level_rmse_among_ties=lv, tied_on_level=near, chosen=chosen)


def run_pool(ex, specs, results, outdir, log):
    futs = {ex.submit(run_job, sp): sp for sp in specs}
    for fu in cf.as_completed(futs):
        sp = futs[fu]
        try:
            r = fu.result()
        except Exception as e:  # broken pool / killed worker
            r = dict(cand=sp["cand"], variant=sp["variant"], job=sp["job"], error="worker failure: " + repr(e)[:200])
        results[(r["cand"], r["variant"], r["job"])] = r
        with open(os.path.join(outdir, "jobs", f"{r['cand']}_{r['variant']}_{r['job']}.json"), "w") as fh:
            json.dump(r, fh)
        if "val_rmse" in r:
            log(f"{r['cand']}/{r['variant']} {r['job']}: val RMSE {r['val_rmse']:.4f}  train loss {r['train_loss']:.5f}  {r['secs']:.0f}s")
        elif "rmse_masked" in r:
            log(f"{r['cand']}/{r['variant']} level: masked RMSE {r['rmse_masked']:.4f}  unmasked {r['rmse_unmasked']:.4f}  {r['secs']:.0f}s")
        else:
            log(f"{r['cand']}/{r['variant']} {r['job']}: NO RESULT {r.get('error') or r.get('reason')}")


class SyncExecutor:
    """debug stand-in for ProcessPoolExecutor: runs each job when it is submitted"""
    def submit(self, fn, *args):
        fu = cf.Future()
        try:
            fu.set_result(fn(*args))
        except Exception as e:
            fu.set_exception(e)
        return fu

    def shutdown(self, *a, **k):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deadline-min", type=float, default=110.0)
    ap.add_argument("--only", default="", help="comma list of candidate ids (debug)")
    ap.add_argument("--maxiter-scale", type=float, default=1.0, help="debug only: scales all iteration counts")
    ap.add_argument("--sync", action="store_true", help="debug only: run the jobs in this process (no pool)")
    ap.add_argument("--stop-after", default="", choices=["", "wave1", "rule1"], help="debug only")
    a = ap.parse_args()
    T0 = time.time()
    deadline = T0 + a.deadline_min * 60
    os.makedirs(os.path.join(a.out, "jobs"), exist_ok=True)
    logf = open(os.path.join(a.out, "select.log"), "a")

    def log(*m):
        s = f"[{now()} +{time.time() - T0:6.0f}s] " + " ".join(str(x) for x in m)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    cand_ids = [c[0] for c in CANDS if not a.only or c[0] in a.only.split(",")]
    sel = dict(started_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(T0)), workers=a.workers, deadline_min=a.deadline_min,
               cv=dict(blocks=BLOCKS, starts=CV_STARTS, maxiter=CV_MAXITER, level_q=LEVEL_Q, level_starts=LEVEL_STARTS))
    results = {}
    ex = SyncExecutor() if a.sync else cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"))
    spec0 = lambda cid, vid, job, warm=None: dict(cand=cid, variant=vid, job=job, warm=warm, deadline=deadline, maxiter_scale=a.maxiter_scale)
    jobs = [f"fold{bi}" for bi in range(4)] + ["level"]

    # ---- stage 1, wave 1: S00 (reference, warm-start source) and S13 (not nested) ------------------------------------
    wave1 = [c for c in ("S00", "S13") if c in cand_ids]
    log("stage 1 wave 1:", wave1)
    run_pool(ex, [spec0(c, "G0", j) for c in wave1 for j in jobs], results, a.out, log)
    t1 = table(results, wave1, "G0")
    if "S00" in t1 and t1["S00"]["complete"]:
        gate = abs(t1["S00"]["cv"] - GATE_CV_C)
        sel["reproduction_gate"] = dict(s00_cv=t1["S00"]["cv"], round1_cv=GATE_CV_C, tol=GATE_CV_TOL, passed=bool(gate <= GATE_CV_TOL),
                                        folds=t1["S00"]["folds"], round1_folds=[0.139, 0.252, 0.177, 0.246], level_rmse=t1["S00"]["level_rmse"], round1_level=0.396)
        log("reproduction gate S00: CV %.4f (round 1: %.4f, tol %.3f) -> %s" % (t1["S00"]["cv"], GATE_CV_C, GATE_CV_TOL, "PASS" if gate <= GATE_CV_TOL else "FAIL"))
        if gate > GATE_CV_TOL and a.maxiter_scale == 1.0:
            sel["abort"] = "reproduction gate failed"
            json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
            log("ABORT: S00 does not reproduce the round-1 pooled CV; nothing else is run")
            ex.shutdown(cancel_futures=True)
            sys.exit(3)
    else:
        sel["abort"] = "S00 incomplete"
        json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
        log("ABORT: S00 incomplete"); ex.shutdown(cancel_futures=True); sys.exit(3)

    if a.stop_after == "wave1":
        log("stop after wave 1 (debug)"); return

    # ---- stage 1, wave 2: nested candidates warm-started from S00's optimum of the same job ----------------------------
    rest = [c for c in cand_ids if c not in wave1]
    specs = []
    for j in jobs:
        zc = results[("S00", "G0", j)]["z"]
        for c in rest:
            specs.append(spec0(c, "G0", j, [warm_from_c(c, zc).tolist()]))
    specs.sort(key=lambda sp: (cand_ids.index(sp["cand"]), sp["job"]))
    log("stage 1 wave 2:", rest)
    run_pool(ex, specs, results, a.out, log)
    rows1 = table(results, cand_ids, "G0")
    chosen1, det1 = apply_rule(rows1, cand_ids)
    sel["stage1"] = dict(rows=rows1, rule=det1, chosen=chosen1)
    log("rule 1 ->", chosen1, json.dumps({k: det1[k] for k in det1 if k in ("best", "best_cv", "se_best", "threshold", "eligible", "tied_on_size", "tied_on_level")}))
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    if chosen1 is None:
        sel["abort"] = "rule 1 found no complete candidate"; json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1); sys.exit(4)
    if a.stop_after == "rule1":
        log("stop after rule 1 (debug)"); return

    # ---- stage 2: fit variants on the chosen structure, warm-started from its own G0 optimum --------------------------
    specs = []
    for vid in ("G3", "G1", "G2"):
        for j in jobs:
            specs.append(spec0(chosen1, vid, j, [results[(chosen1, "G0", j)]["z"]]))
    log("stage 2 on", chosen1, ":", VARIANT_ORDER[1:])
    # speculative dry run of (chosen1, G0) on the free worker slot: it is the answer if rule 2 keeps G0 (the usual case)
    fu_dry0 = ex.submit(dry_job, dict(cand=chosen1, variant="G0", seed=0, maxiter_scale=a.maxiter_scale))
    run_pool(ex, specs, results, a.out, log)
    rows2 = {}
    for vid in VARIANT_ORDER:
        r = table(results, [chosen1], vid)[chosen1]
        rows2[vid] = r
    chosen2, det2 = apply_rule(rows2, VARIANT_ORDER)
    sel["stage2"] = dict(structure=chosen1, rows=rows2, rule=det2, chosen=chosen2)
    log("rule 2 ->", chosen2, json.dumps({k: det2[k] for k in det2 if k in ("best", "best_cv", "se_best", "threshold", "eligible", "tied_on_level")}))
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    if chosen2 is None:
        chosen2 = "G0"

    # ---- dry run on the full estimation record with the final hyper-parameters ------------------------------------------
    log(f"dry run: {chosen1} + {chosen2}, {VARIANTS[chosen2]['final_starts']} starts x {FINAL_MAXITER} iterations, seed 0 (train only)")
    if chosen2 == "G0":
        dry = fu_dry0.result()
    else:
        dry = ex.submit(dry_job, dict(cand=chosen1, variant=chosen2, seed=0, maxiter_scale=a.maxiter_scale)).result()
    log("dry run train RMSE %.5f  (%.0fs)" % (dry["train_rmse"], dry["secs"]))
    ident = dict(compared_with="results/ct_final_ct_phys8.json seed 0", applicable=bool(chosen1 == "S00" and chosen2 in ("G0", "G3")))
    ref_path = os.path.join(ROOT, "results", "ct_final_ct_phys8.json")
    if ident["applicable"] and os.path.exists(ref_path):
        ref = json.load(open(ref_path))["seeds"]["0"]
        rel = {k: abs(dry["theta"][k] - ref["theta"][k]) / max(abs(ref["theta"][k]), 1e-3) for k in BASE}
        ident.update(max_rel_diff=max(rel.values()), rel_diff=rel, train_rmse_round1=ref["train_rmse"], d_train_rmse=abs(dry["train_rmse"] - ref["train_rmse"]))
        ident["identical"] = bool(ident["max_rel_diff"] < IDENT_REL_TOL and ident["d_train_rmse"] < IDENT_TRAIN_TOL)
    else:
        ident["identical"] = False
    sel["final"] = dict(structure=chosen1, free=list(CAND[chosen1][2]), variant=chosen2, variant_spec=VARIANTS[chosen2], final_maxiter=FINAL_MAXITER,
                        dry_run=dry, identical_to_ct_phys8=ident)
    sel["elapsed_s"] = round(time.time() - T0)
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    log("selection done:", chosen1, chosen2, "identical to ct_phys8:", ident["identical"], f"elapsed {time.time() - T0:.0f}s")
    ex.shutdown()


if __name__ == "__main__":
    main()
