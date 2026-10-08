"""Pre-registered paired comparison against the third-party 0.221 code (form fixed BEFORE the frozen run).

For each of our saved test trajectories and each reference-code trajectory (runs/ref_seed*.npz, which cover test[6:]; the common
official window is test[50:], i.e. y_pred[44:] of the reference), compute the paired moving-block-bootstrap (block 25, 4000 draws, the
same resampled time indices for both models) 95% interval of  RMSE_ours - RMSE_ref .  A negative upper bound = 'beats'.
Claim rule (also in the FREEZE json): the headline number is the MEAN over our seeds, never the best seed; 'beats the 0.221 entry'
requires the upper bound < 0 against the seed-42 reference trajectory for every one of our seeds; 'beats the reference code in general'
is reported as the fraction of the other reference seeds against which the upper bound is < 0.
usage: python ct/compare_ref.py <freeze-name>
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402

BLOCK, NBOOT, N0 = 25, 4000, P.N_INIT


def paired_ci(e_a, e_b, seed=0):
    n = len(e_a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / BLOCK))
    starts = np.arange(n - BLOCK + 1)
    d = np.empty(NBOOT)
    a2, b2 = e_a ** 2, e_b ** 2
    for i in range(NBOOT):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        d[i] = np.sqrt(a2[idx].mean()) - np.sqrt(b2[idx].mean())
    pt = float(np.sqrt(a2.mean()) - np.sqrt(b2.mean()))
    return pt, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main(name):
    _, test = P._load_both()
    yt = np.asarray(test.y, float).ravel()
    refs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "ref_seed*.npz"))):
        s = int(os.path.basename(f)[8:-4])
        d = np.load(f)
        yp = d["y_pred"][44:]            # reference covers test[6:]; official window test[50:]
        refs[s] = yp - yt[N0:]
    out = dict(name=name, pairs={})
    mine = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", f"ct_final_{name}_seed*_ypred.npy"))):
        s = int(os.path.basename(f).split("seed")[1].split("_")[0])
        mine[s] = np.load(f)[N0:] - yt[N0:]
    for s, ea in mine.items():
        rm = float(np.sqrt(np.mean(ea ** 2)))
        rows = {}
        for r, eb in refs.items():
            pt, lo, hi = paired_ci(ea, eb)
            rows[str(r)] = dict(ref_rmse=float(np.sqrt(np.mean(eb ** 2))), diff=pt, lo=lo, hi=hi, beats=bool(hi < 0))
        out["pairs"][str(s)] = dict(rmse=rm, vs_ref=rows)
    ok42 = all(v["vs_ref"]["42"]["beats"] for v in out["pairs"].values())
    others = [r for r in refs if r != 42]
    frac = float(np.mean([np.mean([v["vs_ref"][str(r)]["beats"] for v in out["pairs"].values()]) for r in others]))
    out["summary"] = dict(mean_rmse_over_seeds=float(np.mean([v["rmse"] for v in out["pairs"].values()])),
                          beats_0221_entry_every_seed=ok42, fraction_of_other_reference_seeds_beaten=frac,
                          n_our_seeds=len(mine), n_reference_seeds=len(refs))
    json.dump(out, open(os.path.join(ROOT, "results", f"ct_compare_ref_{name}.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    s0 = sorted(out["pairs"], key=int)[0]
    for r in (42, 3, 8):
        if str(r) in out["pairs"][s0]["vs_ref"]:
            v = out["pairs"][s0]["vs_ref"][str(r)]
            print(f"our seed {s0} vs reference seed {r}: ref RMSE {v['ref_rmse']:.4f}  diff {v['diff']:+.4f}  95% [{v['lo']:+.4f}, {v['hi']:+.4f}]")


if __name__ == "__main__":
    main(sys.argv[1])
