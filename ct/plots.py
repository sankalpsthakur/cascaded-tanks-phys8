"""Figures for the technical report (run AFTER the frozen final run; reads saved predictions, selects nothing)."""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402

name = sys.argv[1]
res = json.load(open(os.path.join(ROOT, "results", f"ct_final_{name}.json")))
_, test = P._load_both()
yt = np.asarray(test.y, float).ravel(); ut = np.asarray(test.u, float).ravel()
os.makedirs(os.path.join(ROOT, "results", "figs"), exist_ok=True)
seeds = sorted(res["seeds"], key=int)
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for s in seeds[:10]:
    yp = np.load(os.path.join(ROOT, "runs", f"ct_final_{name}_seed{s}_ypred.npy"))
    ax[0].plot(yp, lw=0.7, alpha=0.6)
ax[0].plot(yt, "k", lw=1.2, label="measured")
ax[0].axvspan(0, P.N_INIT, color="0.9"); ax[0].set_ylabel("y [V]"); ax[0].legend()
yp0 = np.load(os.path.join(ROOT, "runs", f"ct_final_{name}_seed{seeds[0]}_ypred.npy"))
ax[1].plot(yp0 - yt, lw=0.8); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("error seed %s [V]" % seeds[0])
ax[2].plot(ut, lw=0.8); ax[2].set_ylabel("u [V]"); ax[2].set_xlabel("sample (Ts = 4 s)")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "figs", f"ct_{name}_test.png"), dpi=110)
# seed distribution vs reference
ref = [json.load(open(f))["rmse_official_ninit50"] for f in sorted(glob.glob(os.path.join(ROOT, "results", "ref_repro_seed*.json")))]
mine = [res["seeds"][s]["test_rmse"] for s in seeds]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.boxplot([ref, mine], labels=["0.221-entry code\n(11 seeds)", f"{name}\n({len(mine)} seeds)"], vert=False)
ax.axvline(0.221, color="r", ls="--", lw=1); ax.set_xlabel("test RMSE [V], official n_init=50")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "figs", f"ct_{name}_seeds.png"), dpi=110)
print("saved figures")
