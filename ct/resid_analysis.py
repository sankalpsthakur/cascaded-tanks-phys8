"""Train-only residual analysis of a CV structure: where (which regime) do held-out-block errors come from?"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P, phys  # noqa: E402
from ct.cv import BLOCKS, N_INIT  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "C0 base(+b4+cA)"
recs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "cv_round2.jsonl"))]
rec = [r for r in recs if r["name"] == name][-1]
u, y, _ = P.load_train()
BASE = ("k1", "r2", "k3", "k4", "beta", "a", "b4", "cA")
st = phys.Structure(free=BASE, tie_r2=False)
f = phys.Fitter(st)
rows = []
for fo in rec["folds"]:
    s, e = BLOCKS[fo["block"]]
    z = np.array(fo["z"])
    ub, yb = u[s:e], y[s:e]
    uh = u[s - 2:s] if s >= 2 else np.array([u[0], u[0]])
    zx, _ = f.fit_x0(z, ub[:N_INIT], yb[:N_INIT], uh)
    zz = np.concatenate([z[:f.nfree], zx])
    th, x0 = phys.unpack(st, zz[:f.nfree], zz[f.nfree:])
    yh, xs = st and f.sim(th, x0, ub, uh)
    yh, xs = np.asarray(yh), np.asarray(xs)
    res = (yh - yb)
    reg = np.full(len(yb), "normal", dtype=object)
    reg[xs[:, 0] >= phys.XM1 - 1e-6] = "upper-overflow"
    reg[yb >= 9.999] = "sat-y=10"
    for lab in ("normal", "upper-overflow", "sat-y=10"):
        m = (reg == lab) & (np.arange(len(yb)) >= N_INIT)
        if m.sum():
            rows.append((fo["block"], lab, int(m.sum()), float(np.sqrt(np.mean(res[m] ** 2))), float(np.mean(res[m]))))
    # contiguous high-error stretches
    big = np.where(np.abs(res[N_INIT:]) > 0.35)[0] + N_INIT
    print(f"fold {fo['block']} val {fo['val_rmse']:.3f}: |res|>0.35 at block-idx", (big[:60].tolist()), "..." if len(big) > 60 else "")
print("fold regime n rmse bias")
for r in rows:
    print(r[0], r[1], r[2], round(r[3], 3), round(r[4], 3))
