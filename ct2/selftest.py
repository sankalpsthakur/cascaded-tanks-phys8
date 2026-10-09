"""Fit-free self-tests of ct2/phys2.py (forward evaluations, gradient checks, loss plumbing). Touches the estimation record only.
usage: python ct2/selftest.py <test-name> ...   (run each under bin/guard.sh: 60 s, 512 MiB)
  equiv-C | equiv-F1 | equiv-H1 | equiv-qov | equiv-c4 | equiv-cA1 | equiv-E | equiv-D   same trajectory as ct/phys.py (frozen) for the shared terms
  limits   tp, to -> 0 and w3 = 0 reduce to the base model; tp large freezes the lagged input
  grad_to | grad_mse | grad_huber | grad_ms   JAX gradient vs central differences for the new terms (tp, to, w3), the Huber loss and the multiple-shooting loss
  ms       multiple-shooting loss with SS start states: gaps vanish and the data term equals the single-shooting loss
  final_dummy   final2.py end to end with the TRAIN record as a stand-in for the test record (no test value read)
  plumb_base | plumb_new   two-iteration fit of the base structure / a +tp +w3 structure on 160 samples (plumbing only, discarded)
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P, phys  # noqa: E402  (frozen round-1 module)
from ct2 import phys2  # noqa: E402

BASE = ("k1", "r2", "k3", "k4", "beta", "a", "b4", "cA")
Z_BASE = np.array([np.log(0.159), np.log(0.725), np.log(0.126), np.log(0.251), np.log(0.64), 1.81, 1.49, -0.173])
EXTRA_Z = {"p1": np.log(0.52), "p3": np.log(0.47), "c2": 0.004, "c4": 0.01, "cA1": -0.03, "h0": 0.6, "qov": np.log(1.3), "d1": -1.5, "d2": -3.0,
           "tp": np.log(1.2), "to": np.log(0.8), "w3": 0.25}


def data(n=160):
    u, y, _ = P.load_train()
    return u[:n], y[:n]


def zvec(free):
    d = dict(zip(BASE, Z_BASE)); d.update(EXTRA_Z)
    free = phys2.Structure(free=free).free          # z layout = order of PARAMS (what unpack() uses in both modules)
    return np.array([d[n] for n in free] + [np.log(4.0), np.log(2.0)])


def equiv(extra):
    u, y = data()
    free = BASE + tuple(extra)
    z = zvec(free)
    uh = np.array([u[0], u[0]])
    f1 = phys.Fitter(phys.Structure(free=free, tie_r2=False))
    f2 = phys2.Fitter(phys2.Structure(free=free))
    a, b = np.asarray(f1.simj(z, u, uh)), np.asarray(f2.simj(z, u, uh))
    err = float(np.max(np.abs(a - b)))
    print(f"equiv {extra or 'C'}: max |ct.phys - ct2.phys2| = {err:.3e} (range of y {a.min():.2f}..{a.max():.2f})")
    assert err < 1e-9, err


def limits():
    u, y = data(192)
    uh = np.array([u[0], u[0]])
    f0 = phys2.Fitter(phys2.Structure(free=BASE))
    z0 = zvec(BASE)
    y0 = np.asarray(f0.simj(z0, u, uh))
    for extra, over, tol in ((("w3",), {"w3": 0.0}, 1e-12), (("tp",), {"tp": np.log(1e-4)}, 2e-3), (("to",), {"to": np.log(1e-5)}, 1e-6)):
        free = BASE + extra
        z = zvec(free)
        for k, v in over.items():
            z[phys2.Structure(free=free).free.index(k)] = v
        f = phys2.Fitter(phys2.Structure(free=free))
        yy = np.asarray(f.simj(z, u, uh))
        print(f"limit {extra}: max |y - y_base| = {np.max(np.abs(yy - y0)):.3e}")
        assert np.max(np.abs(yy - y0)) < tol
    # tp huge: the lagged input stays at u_hist[1]  ==  base model driven by the constant u_hist[1]
    free = BASE + ("tp",)
    z = zvec(free); z[phys2.Structure(free=free).free.index("tp")] = np.log(1e9)
    yy = np.asarray(phys2.Fitter(phys2.Structure(free=free)).simj(z, u, uh))
    yc = np.asarray(f0.simj(z0, np.full_like(u, u[0]), uh))
    print(f"limit tp=1e9: max |y - y_base(const u)| = {np.max(np.abs(yy - yc)):.3e}")
    assert np.max(np.abs(yy - yc)) < 1e-6


def fd_check(fitter, z, u, y, w, uh, idx, eps=1e-6):
    v, g = fitter.vg(z, u, y, w, uh)
    g = np.asarray(g)
    out = []
    for i in idx:
        zp, zm = z.copy(), z.copy(); zp[i] += eps; zm[i] -= eps
        fd = (float(fitter.loss(zp, u, y, w, uh)) - float(fitter.loss(zm, u, y, w, uh))) / (2 * eps)
        out.append((i, g[i], fd))
    return out


def _grad(kind):
    u, y = data(96)
    w = np.ones_like(y); w[30:45] = 0.0
    uh = np.array([u[0], u[0]])
    free = BASE if kind == "ms" else BASE + ("tp", "to", "w3")
    z = zvec(free)
    if kind == "ms":
        f = phys2.Fitter(phys2.Structure(free=free), ms_K=4, ms_lam=30.0)
        z = f.extend_ms(z, u, uh)
        z[len(free) + 2:] += 0.01          # break continuity so that the penalty has a gradient
        idx = [0, 3, len(free) - 1, len(free) + 2, len(free) + 5, len(free) + 7]
    else:
        f = phys2.Fitter(phys2.Structure(free=free), **({"loss": "huber", "huber_delta": 0.2} if kind == "huber" else {}))
        idx = list(range(len(free) + 2))
    for i, g, fd in fd_check(f, z, u, y, w, uh, idx):
        print(f"grad {kind} z[{i}] jax {g:+.6e} fd {fd:+.6e}")
        assert abs(g - fd) < 1e-5 * max(1.0, abs(fd)) + 1e-7, (kind, i, g, fd)


def grad_to():
    """gradient of the delayed-overflow term on a record that contains overflow (first 170 samples)"""
    u, y = data(170)
    w = np.ones_like(y); w[60:75] = 0.0
    uh = np.array([u[0], u[0]])
    free = BASE + ("to",)
    z = zvec(free)
    f = phys2.Fitter(phys2.Structure(free=free))
    res = fd_check(f, z, u, y, w, uh, [4, 8, 9])
    for i, g, fd in res:
        print(f"grad to z[{i}] jax {g:+.6e} fd {fd:+.6e}")
        assert abs(g - fd) < 1e-5 * max(1.0, abs(fd)) + 1e-7, (i, g, fd)
    assert abs(res[0][1]) > 1e-6 and abs(res[2][1]) > 1e-6, "overflow not exercised"


def grad_mse(): _grad("mse")
def grad_huber(): _grad("huber")
def grad_ms(): _grad("ms")


def ms():
    u, y = data(128)
    w = np.ones_like(y)
    uh = np.array([u[0], u[0]])
    f_ss = phys2.Fitter(phys2.Structure(free=BASE))
    f_ms = phys2.Fitter(phys2.Structure(free=BASE), ms_K=4, ms_lam=30.0)
    z = zvec(BASE)
    l_ss = float(f_ss.loss(z, u, y, w, uh))
    zm = f_ms.extend_ms(z, u, uh)
    l_ms = float(f_ms.loss(zm, u, y, w, uh))
    print(f"ms: single-shooting loss {l_ss:.10f}  multiple-shooting loss at SS start states {l_ms:.10f}")
    assert abs(l_ss - l_ms) < 1e-9


def _plumb(free):
    u, y = data(160)
    uh = np.array([u[0], u[0]])
    f = phys2.Fitter(phys2.Structure(free=free))
    z, loss, diag = f.fit(u, y, n_starts=1, seed=0, maxiter=2, warm=[zvec(free)])
    zx, wl = f.fit_x0(z, u[:50], y[:50], uh, n_starts=2)
    yh = f.simulate_with_init(z, zx, u, uh)
    print(f"plumb {len(free)} params: loss {loss:.4f} diag {[(d['kind'], round(d.get('loss', -1), 4)) for d in diag]} window {wl:.4f} rmse {phys2.block_rmse(yh, y, 50):.3f}")
    assert np.all(np.isfinite(yh))


def plumb_base(): _plumb(BASE)
def plumb_new(): _plumb(BASE + ("tp", "w3"))


def warm(cid):
    """loss of the nested candidate at the warm vector built from the round-1 optimum of C (new terms off) vs the loss of C itself"""
    import json
    from ct2 import select2
    u, y, _ = P.load_train()
    th = json.load(open(os.path.join(ROOT, "results", "ct_final_ct_phys8.json")))["seeds"]["0"]["theta"]
    zc = np.array([np.log(th[n]) if phys2.PARAMS[n][0] == "pos" else th[n] for n in phys2.Structure(free=BASE).free] + [np.log(4.0), np.log(2.0)])
    uh = np.array([u[0], u[0]]); w = np.ones_like(y)
    lc = float(select2.make_fitter("S00", "G0").loss(zc, u, y, w, uh))
    if cid == "S00":
        print(f"warm S00: loss {lc:.6f}"); return
    zw = select2.warm_from_c(cid, zc)
    f = select2.make_fitter(cid, "G0")
    assert len(zw) == f.nfree + 2, (len(zw), f.nfree)
    lw = float(f.loss(zw, u, y, w, uh))
    print(f"warm {cid}: loss at warm vector {lw:.6f} vs C {lc:.6f}  (rel {abs(lw - lc) / lc:.2e})")
    assert abs(lw - lc) / lc < 5e-3


def job(spec):
    """run_job plumbing for '<cand>:<variant>:<job>' (e.g. S12:G0:fold0 or S12:G2:level) with 2 iterations per start; the result is discarded"""
    import json, time
    from ct2 import select2
    cid, vid, jb = spec.split(":")
    u, y, _ = P.load_train()
    zc = np.array([np.log(0.159), np.log(0.725), np.log(0.126), np.log(0.251), np.log(0.64), 1.81, 1.49, -0.173, np.log(4.0), np.log(2.0)])
    warm = [select2.warm_from_c(cid, zc).tolist()] if select2.CAND[cid][3] else []
    if vid != "G0":
        warm = [select2.warm_from_c(cid, zc).tolist()] if select2.CAND[cid][3] else []
    t0 = time.time()
    r = select2.run_job(dict(cand=cid, variant=vid, job=jb, warm=warm, deadline=time.time() + 600, maxiter_scale=0.005))
    print(f"job {spec}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("val_rmse", "rmse_masked", "rmse_unmasked", "train_loss", "error", "secs", "n_numbers")}), f"wall {time.time() - t0:.0f}s")
    print("starts", [(d["kind"], round(d.get("loss", -1), 4), d.get("nit")) for d in r.get("starts", [])])
    assert "error" not in r, r.get("tb")


def final_dummy():
    """ct2/final2.py end to end (fit, gate, scoring, TEST-LOG-CT2 row, outputs) with the TRAIN record standing in for the test record:
    no test value is read. 1 seed, 2 iterations, outputs in a scratch directory."""
    import json, tempfile
    from nonlinear_benchmarks.utilities import Input_output_data
    from ct2 import final2
    orig = P._load_both

    def fake():
        train, _ = orig()                      # the real test arrays are dropped here, never touched
        t = Input_output_data(u=train.u, y=train.y, sampling_time=train.sampling_time, name="stand-in test = train")
        t.state_initialization_window_length = 50
        return train, t

    P._load_both = fake
    tmp = tempfile.mkdtemp(prefix="ct2_final_dummy_")
    os.makedirs(os.path.join(tmp, "runs")); os.makedirs(os.path.join(tmp, "results"))
    final2.ROOT = tmp
    final2.LOG2 = os.path.join(tmp, "TEST-LOG-CT2.md")
    final2.tree_clean_and_committed = lambda p: (True, True, "")
    cfg = dict(name="dummy", structure=dict(free=list(BASE), clamp_x2=True, nsub=4), fit=dict(loss="mse", n_starts=1, maxiter=2),
               gate=dict(train_rmse_seed0=0.9, tol=10.0), identical_to_ct_phys8=False)
    fp = os.path.join(tmp, "FREEZE_dummy.json")
    json.dump(cfg, open(fp, "w"))
    sys.argv = ["final2.py", fp, "--seeds", "0", "--workers", "0"]
    final2.main()
    print(open(final2.LOG2).read()[-420:])
    r = json.load(open(os.path.join(tmp, "results", "ct_final_dummy.json")))
    assert r["summary"]["n"] == 1 and os.path.exists(os.path.join(tmp, "runs", "ct_final_dummy_seed0_ypred.npy"))
    assert open(final2.LOG2).read().count("| 25 |") == 1


def rules():
    """rule arithmetic on the round-1 fold numbers of C (pooled 0.2089, SE ~0.026) and synthetic competitors"""
    from ct2 import select2
    pooled, se = select2.pooled_and_se([0.139, 0.252, 0.177, 0.246])
    print(f"C: pooled {pooled:.4f}  SE {se:.4f}")
    assert abs(pooled - 0.2089) < 5e-4 and abs(se - 0.026) < 1e-3
    mk = lambda cid, folds, lv, n: dict(cand=cid, complete=True, folds=folds, level_rmse=lv, n_numbers=n, **dict(zip(("cv", "se"), select2.pooled_and_se(folds))))
    rows = {"S00": mk("S00", [0.139, 0.252, 0.177, 0.246], 0.396, 10), "S01": mk("S01", [0.139, 0.245, 0.17, 0.24], 0.30, 11), "S13": mk("S13", [0.14, 0.24, 0.18, 0.25], 0.30, 10)}
    order = ["S00", "S01", "S13"]
    ch, det = select2.apply_rule(rows, order)
    print("choice", ch, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in det.items()})
    assert ch == "S13", ch          # S01 best but C and S13 within 1 SE; both 10 numbers; S13 has the lower level RMSE
    rows["S13"]["level_rmse"] = 0.398
    ch, _ = select2.apply_rule(rows, order)
    assert ch == "S00", ch          # level RMSEs within 0.005 -> lower index (S00)
    rows["S01"] = mk("S01", [0.13, 0.20, 0.17, 0.19], 0.30, 11)   # a clear improvement (C falls outside best + 1 SE) displaces C and S13
    ch, det = select2.apply_rule(rows, order)
    print("clear improvement ->", ch, round(det["threshold"], 4))
    assert ch == "S01", ch


if __name__ == "__main__":
    for t in sys.argv[1:]:
        if t.startswith("warm-"):
            warm(t[5:])
        elif t.startswith("job-"):
            job(t[4:])
        elif t.startswith("equiv-"):
            name = t[6:]
            equiv({"C": (), "F1": ("d1", "d2"), "H1": ("h0",), "qov": ("qov",), "c4": ("c4",), "cA1": ("cA1",), "E": ("p1", "p3"), "D": ("c2",)}[name])
        else:
            globals()[t]()
    print("OK")
